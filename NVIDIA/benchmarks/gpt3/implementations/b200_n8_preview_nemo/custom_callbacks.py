# Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import time
from argparse import Namespace
from datetime import datetime

import math
from itertools import repeat
from pathlib import Path
from typing import Dict, Any, Optional, Union

import pytorch_lightning
import pytorch_lightning as pl
import torch
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core import parallel_state
from megatron.core.datasets.gpt_dataset import _get_ltor_masks_and_position_ids

import nemo
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import \
    MegatronPretrainingSampler
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import MockGPTDataset
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import \
    BaseMegatronBatchSampler
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import \
    MegatronGPTModel
from nemo.constants import NEMO_ENV_VARNAME_TESTING
from nemo.utils import AppState
from nemo.utils.env_var_parsing import get_envbool
from nemo.utils.exp_manager import TimingCallback, \
    SkipResumeTrainingValidationLoop
from nemo.utils.formatters.base import BaseNeMoFormatter, DebugNeMoFormatter
from nemo.utils.get_rank import is_global_rank_zero
from nemo.utils.timers import NamedTimer
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loops import _TrainingEpochLoop as TrainingEpochLoop
from pytorch_lightning.utilities import rank_zero_only
from lightning_fabric.utilities.cloud_io import get_filesystem
from torch.utils.data import default_collate

from mlperf_logger import mllogger
from transformer_engine.common import recipe
from transformer_engine.pytorch import make_graphed_callables

_PATH = Union[str, Path]

logger = logging.getLogger(__name__)

def compute_consumed_mllog_tokens(trainer, model):
    steps_since_resume = trainer.global_step - model.init_global_step
    gbs = model.cfg.global_batch_size
    model_gbs = AppState().data_parallel_size * model.cfg.micro_batch_size * get_num_microbatches()
    assert gbs == model_gbs, (gbs, model_gbs)
    consumed_samples = (
        steps_since_resume * gbs
    )
    return int(consumed_samples) * model.cfg.data.seq_length


def run_training_warmup(trainer, warmup_train_steps, warmup_validation_steps):
    torch.distributed.barrier()
    start = time.time()

    # Warmup for training
    # Run forward and backward (no optimizer step)
    logger.info(f'Starting training warmup')
    for i in range(warmup_train_steps):
        trainer.model.training_step(trainer.model.get_synthetic_input_training())

    torch.distributed.barrier()
    logger.info(f'Finished training warmup: {time.time() - start}s. Starting validation warmup')
    # Warmup for validation
    if warmup_validation_steps > 0:
        #trainer.validating = True # Causes issues with eval accuracy
        trainer.testing = True
        trainer.training = not trainer.testing
        trainer.model.set_training(trainer.training)
        for i in range(warmup_validation_steps):
            trainer.model.validation_step(trainer.model.get_synthetic_input_validation())
        #trainer.validating = False
        trainer.testing = False
        trainer.training = not trainer.testing
        trainer.model.set_training(trainer.training)

    # For GPT `zero_grad` is a noop, but included here for completeness
    trainer.model.zero_grad()
    trainer._logger_connector.reset_results()
    trainer._logger_connector.reset_metrics()
    torch.distributed.barrier()
    logger.info(f'Time spent in run_training_warmup: {time.time() - start}s')

def get_microbatch_schedule(num_microbatches, num_model_chunks):
    schedule = []
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()

    def get_model_chunk_id(microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = microbatch_id % (pipeline_model_parallel_size * num_model_chunks)
        model_chunk_id = microbatch_id_in_group // pipeline_model_parallel_size
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    if pipeline_model_parallel_size > 1:
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            #forward_backward_pipelining_with_interleaving
            total_num_microbatches = num_microbatches * num_model_chunks
            if num_microbatches == pipeline_model_parallel_size:
                num_warmup_microbatches = total_num_microbatches
            else:
                num_warmup_microbatches = (pipeline_model_parallel_size - pipeline_parallel_rank - 1) * 2
                num_warmup_microbatches += (num_model_chunks - 1) * pipeline_model_parallel_size
                num_warmup_microbatches = min(num_warmup_microbatches, total_num_microbatches)
            num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches
            for k in range(num_warmup_microbatches):
                cur_model_chunk_id = get_model_chunk_id(k, forward=True)
                schedule.append(cur_model_chunk_id+1)
            for k in range(num_microbatches_remaining):
                forward_k = k + num_warmup_microbatches
                cur_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
                schedule.append(cur_model_chunk_id+1)
                backward_model_chunk_id = get_model_chunk_id(k, forward=False)
                schedule.append(-backward_model_chunk_id-1)
            for k in range(num_microbatches_remaining, total_num_microbatches):
                backward_model_chunk_id = get_model_chunk_id(k, forward=False)
                schedule.append(-backward_model_chunk_id-1)
        else:
            #forward_backward_pipelining_without_interleaving
            num_warmup_microbatches = (
                pipeline_model_parallel_size
                - pipeline_parallel_rank
                - 1
            )
            num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
            num_microbatches_remaining = num_microbatches - num_warmup_microbatches
            schedule = [1]*num_warmup_microbatches + [1,-1]*num_microbatches_remaining + [-1]*num_warmup_microbatches
    else:
        #forward_backward_no_pipelining
        schedule = [1, -1]
    return schedule

def get_num_graphs(schedule):
    model_chunks = max(schedule)
    num_graphs = [0]*model_chunks
    for i in range(model_chunks):
        chunk = i+1
        for c in schedule:
            if c == chunk:
                num_graphs[i] += 1
            if c == -chunk:
                break
    return num_graphs

def run_training_cudagraph(trainer, cfg):
    # Function to perform CUDA graph capture for decoder layers optionally,
    # then perform warmup iterations for training and validation. CUDA graph
    # is currently supported only for training.
    torch.cuda.synchronize()
    torch.distributed.barrier()
    logger.info(f'In run_training_cudagraph')
    start = time.time()
    trainer.model._optimizer.zero_grad()
    torch.distributed.barrier()
    torch.cuda.set_stream(torch.cuda.default_stream())
    schedule = get_microbatch_schedule(get_num_microbatches(), len(trainer.model.model) if isinstance(trainer.model.model, list) else None)
    num_graphs = get_num_graphs(schedule)
    callables = []
    model_chunks = trainer.model.model if isinstance(trainer.model.model, list) else [trainer.model.model]

    # CUDA graph capture
    if cfg.external_cuda_graph:
        torch.distributed.barrier()
        logger.info(f'Starting CUDA graph capture')
        if trainer.model.initialize_ub:
            trainer.model.initialize_ub_func()
        device = trainer.model.model[0].module.decoder.layers[0].layernorm_mlp.fc1_weight.device if isinstance(trainer.model.model, list) else trainer.model.model.module.decoder.layers[0].layernorm_mlp.fc1_weight.device
        sequence_parallel = cfg.sequence_parallel
        tensor_model_parallel_size = cfg.tensor_model_parallel_size
        micro_batch_size = cfg.micro_batch_size
        slen = cfg.encoder_seq_length // tensor_model_parallel_size if sequence_parallel else cfg.encoder_seq_length
        pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
        pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
        if os.getenv('CGDEBUG','0') == '1' and torch.distributed.get_rank() % (torch.distributed.get_world_size() / pipeline_parallel_size) == 0:
           print (f'SCHEDULE {schedule} num_graphs {num_graphs}')
        sample_args = []
        if cfg.fp8_e4m3:
            fp8_format = recipe.Format.E4M3
        elif cfg.fp8_hybrid:
            fp8_format = recipe.Format.HYBRID
        else:
            raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")
        fp8_recipe = recipe.DelayedScaling(
            margin=cfg.fp8_margin,
            interval=cfg.fp8_interval,
            fp8_format=fp8_format,
            amax_compute_algo=cfg.fp8_amax_compute_algo,
            amax_history_len=cfg.fp8_amax_history_len,
            override_linear_precision=(False, False, False),
        )
        if os.getenv('AMORTIZED_CG', '0') == '1' or ((pipeline_parallel_size > 1) and (get_num_microbatches() > pipeline_parallel_size)):
            for b in range(get_num_microbatches()):
                for m_no, model in enumerate(model_chunks):
                    new_model_chunks = []
                    callables = []
                    sample_args = []

                    num_graphed_microbatches = num_graphs[m_no] if num_graphs[m_no] == get_num_microbatches() else num_graphs[m_no]+1
                    if b < num_graphed_microbatches:
                        # Collect layers for CUDA graph capture
                        new_model_chunks.append(model)
                        for l_no, layer in enumerate(model.module.decoder.layers):
                            callables.append(layer)
                            graph_input = (torch.ones((slen, micro_batch_size, cfg.hidden_size), dtype=torch.bfloat16, requires_grad=True, device=device),)
                            sample_args.append(graph_input)

                    if len(callables) > 0:
                        new_schedule = []
                        for m_no, model in enumerate(new_model_chunks):
                            new_schedule.insert(m_no, m_no+1)
                            new_schedule.insert(m_no+1, -m_no-1)
                        graphs = make_graphed_callables(tuple(callables), tuple(sample_args), _order=new_schedule, allow_unused_input=True, fp8_enabled=cfg.fp8, fp8_recipe=fp8_recipe if cfg.fp8 else None, fp8_weight_caching=True, num_warmup_iters=3 if b==0 else 0,)
                        for m_no, model in enumerate(new_model_chunks):
                            for l_no, layer in enumerate(model.module.decoder.layers):
                                if l_no not in model.module.decoder.cuda_graphs:
                                    model.module.decoder.cuda_graphs[l_no] = []
                                model.module.decoder.cuda_graphs[l_no].append(graphs[m_no * len(model.module.decoder.layers) + l_no])
            for m_no, model in enumerate(model_chunks):
                num_graphed_microbatches = num_graphs[m_no] if num_graphs[m_no] == get_num_microbatches() else num_graphs[m_no]+1
                for l_no, layer in enumerate(model.module.decoder.layers):
                    assert len(model.module.decoder.cuda_graphs[l_no]) == num_graphed_microbatches
        else:
            # Collect layers for CUDA graph capture
            for m_no, model in enumerate(model_chunks):
                for l_no, layer in enumerate(model.module.decoder.layers):
                    callables.append(layer)
                    for b in range(num_graphs[m_no]):
                        graph_input = (torch.ones((slen, micro_batch_size, cfg.hidden_size), dtype=torch.bfloat16, requires_grad=True, device=device),)
                        sample_args.append(graph_input)
            graphs = make_graphed_callables(tuple(callables), tuple(sample_args), _order=schedule, allow_unused_input=True, fp8_enabled=cfg.fp8, fp8_recipe=fp8_recipe if cfg.fp8 else None, fp8_weight_caching=True)
            for m_no, model in enumerate(model_chunks):
                for l_no, layer in enumerate(model.module.decoder.layers):
                    model.module.decoder.cuda_graphs[l_no] = []
                    for b in range(num_graphs[m_no]):
                        model.module.decoder.cuda_graphs[l_no].append(graphs[m_no * num_graphs[m_no] * len(model.module.decoder.layers) + b * len(model.module.decoder.layers) + l_no])
        trainer.model.initialize_ub = False

    # Warmup for training
    # Run forward and backward (no optimizer step)
    torch.distributed.barrier()
    logger.info(f'Starting training warmup')
    for i in range(cfg.custom.warmup_train_steps):
        trainer.model.training_step(trainer.model.get_synthetic_input_training())

    torch.distributed.barrier()
    logger.info(f'Finished training warmup: {time.time() - start}s. Starting validation warmup')
    # Warmup for validation
    if cfg.custom.warmup_validation_steps > 0:
        trainer.testing = True
        trainer.training = not trainer.testing
        trainer.model.set_training(trainer.training)
        for i in range(cfg.custom.warmup_validation_steps):
            trainer.model.validation_step(trainer.model.get_synthetic_input_validation())
        trainer.testing = False
        trainer.training = not trainer.testing
        trainer.model.set_training(trainer.training)

    for module in tuple(callables):
        for m in module.modules():
            if hasattr(m, 'reset_fp8_meta_tensors'):
                m.reset_fp8_meta_tensors()

    trainer.model.zero_grad()
    trainer._logger_connector.reset_results()
    trainer._logger_connector.reset_metrics()
    torch.distributed.barrier()
    logger.info(f'Time spent in run_training_cudagraph: {time.time() - start}s')


def reset_fp8_state(model):
    """ Sets `fp8_initialized` flag to False in every TE layer which will force reinitialization. """
    logger.info('Forcing FP8 stats reinitialization')

    def reset_fp8(m):
        if hasattr(m, 'fp8_initialized'):
            m.fp8_initialized = False

    models = model.model
    for model in models if isinstance(models, list) else [models]:
         model.apply(reset_fp8)

class CustomCallback(Callback):
    def __init__(self, cfg):
        super().__init__()
        if cfg.model.custom.force_success_status:
            self.status = mllogger.constants.SUCCESS
        else:
            self.status = mllogger.constants.ABORTED
        self.is_target_reached = False
        self.is_run_stop_already_logged = False

        self.tokens_per_block = cfg.trainer.val_check_interval * cfg.model.global_batch_size * cfg.model.encoder_seq_length
        self.iter_after_valid = False

    def set_success_status(self):
        self.status = mllogger.constants.SUCCESS
        self.is_target_reached = True

    @rank_zero_only
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        mllogger.start(key=mllogger.constants.EPOCH_START,
                       metadata={'epoch_num': compute_consumed_mllog_tokens(trainer, pl_module)}, sync=False)
        mllogger.start(key=mllogger.constants.BLOCK_START,
                       metadata={'first_epoch_num': compute_consumed_mllog_tokens(trainer, pl_module),
                                 'epoch_count': self.tokens_per_block},
                       sync=False)

        return super().on_train_epoch_start(trainer, pl_module)

    @rank_zero_only
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        mllogger.end(key=mllogger.constants.EPOCH_STOP,
                     metadata={'epoch_num': compute_consumed_mllog_tokens(trainer, pl_module)}, sync=False)
        return super().on_train_epoch_end(trainer, pl_module)

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.maybe_log_run_stop(trainer, pl_module)
        return super().on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        mllogger.end(key=mllogger.constants.BLOCK_STOP,
                     metadata={'first_epoch_num': compute_consumed_mllog_tokens(trainer, pl_module) - self.tokens_per_block,
                               'epoch_count': self.tokens_per_block},
                     sync=False)
        mllogger.start(key=mllogger.constants.EVAL_START,
                       metadata={'epoch_num': compute_consumed_mllog_tokens(trainer, pl_module)}, sync=False)
        return super().on_validation_start(trainer, pl_module)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        mllogger.end(key=mllogger.constants.EVAL_STOP,
                     metadata=dict(epoch_num=compute_consumed_mllog_tokens(trainer, pl_module)), sync=False)
        if self.is_target_reached:
            self.maybe_log_run_stop(trainer, pl_module)
        self.iter_after_valid = True
        if bool(os.getenv('ENABLE_EVAL_BARRIER','')):
            torch.cuda.synchronize()
            torch.distributed.barrier()
            logger.info(f'Validation End')
        return super().on_validation_end(trainer, pl_module)

    @rank_zero_only
    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int) -> None:
        if self.iter_after_valid:
            mllogger.start(key=mllogger.constants.BLOCK_START,
                        metadata={'first_epoch_num': compute_consumed_mllog_tokens(trainer, pl_module),
                                    'epoch_count': self.tokens_per_block},
                        sync=False)
            self.iter_after_valid = False

    @rank_zero_only
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        print(f":::MLLOG Weight initialization: {state_dict.keys()}")
        return super().load_state_dict(state_dict)

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if pl_module.cfg.custom.run_warmup_on_synth_data:
            run_training_cudagraph(trainer, pl_module.cfg)
            if pl_module.cfg.fp8 and pl_module.cfg.custom.reset_fp8_stats_after_warmup:
                reset_fp8_state(pl_module)

        # HACK for accurate logging
        for callback in trainer.callbacks:
            if isinstance(callback, DeltaTimingCallback):
                callback.trigger_tick()

        # Note: run on all ranks (to allow synchronization)
        mllogger.log_init_stop_run_start()

    def maybe_log_run_stop(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # Note: run on all ranks (to allow synchronization)
        if self.is_run_stop_already_logged:
            return

        mllogger.end(key=mllogger.constants.RUN_STOP, unique=True, sync=True,
                     metadata=dict(status=self.status))
        mllogger.event(key="trained_samples",
                       value=compute_consumed_mllog_tokens(trainer, pl_module),
                       unique=True, sync=False)
        mllogger.event(key="train_samples",
                       value=compute_consumed_mllog_tokens(trainer, pl_module),
                       unique=True, sync=False)
        self.is_run_stop_already_logged = True


class MetricsLogger(Logger):
    def __init__(self, trainer, model, custom_callback, target_val_log_ppl,
                 extend_run_evals=0,
                 train_loss_key='reduced_train_loss', val_loss_key='val_loss',
                 timing_keys=('train_step_timing', 'train_epoch_timing', 'validation_step_timing', 'validation_epoch_timing'),
                 throughput_key='train_epoch_timing'):
        super().__init__()
        self.trainer = trainer
        self.model = model
        self.custom_callback = custom_callback
        self.target_val_log_ppl = target_val_log_ppl
        self.val_loss_key = val_loss_key
        self.train_loss_key = train_loss_key
        self.timing_keys = [s + ' in s' for s in timing_keys]
        self.throughput_key = throughput_key

        self.extend_run_evals = extend_run_evals
        self.extension_eval_idx = 0
        self.is_target_reached = False


    def log_metrics(self, metrics: Dict[str, float],
                    step: Optional[int] = None) -> None:
        if self.val_loss_key in metrics:
            self._log_val_metrics(metrics, step)
        self._log_throughputs(metrics, step)
        if bool(os.getenv('ENABLE_TRAIN_BARRIER','')):
            torch.cuda.synchronize()
            torch.distributed.barrier()
            if bool(os.getenv('LOG_TRAIN_BARRIER','')):
                logger.info(f'Train Step End')
        # Consumed samples is shifted by 1 (in terms of gbs), beacuse `trainer.global_step`
        # is not incremented by the time `consumed_samples` is logged (in model forward)
        # Recomputing in here:
        if 'consumed_samples' in self.trainer.callback_metrics:
            correct_consumed_samples = self.model.compute_consumed_samples(self.trainer.global_step - self.model.init_global_step)
            self.trainer.callback_metrics['consumed_samples'].fill_(correct_consumed_samples)


    def _log_val_metrics(self, metrics: Dict[str, float],
                         step: Optional[int] = None):
        assert self.val_loss_key in metrics, metrics.keys()
        val_loss = metrics[self.val_loss_key]
        val_ppl = math.exp(min(20, val_loss))
        mllogger.event(mllogger.constants.EVAL_ACCURACY, value=val_loss,
                       metadata=dict(epoch_num=compute_consumed_mllog_tokens(self.trainer, self.model)))

        if not self.is_target_reached and val_loss <= self.target_val_log_ppl:
            logger.info(f'Target Log PPL {self.target_val_log_ppl} reached')
            self.custom_callback.set_success_status()
            self.is_target_reached = True
            if self.extend_run_evals:
                logger.info(f'Continuing training for {self.extend_run_evals} extra eval intervals')
            else:
                logger.info(f'Stopping training after reaching target log PPL')
                self.trainer.should_stop = True

        if self.is_target_reached and self.extend_run_evals:
            if self.extension_eval_idx >= self.extend_run_evals:
                logger.info(f'Stopping training after {self.extend_run_evals} extra eval intervals')
                self.trainer.should_stop = True
            self.extension_eval_idx += 1

    def _log_throughputs(self, metrics: Dict[str, float],
                         step: Optional[int] = None):

        for timing_key in self.timing_keys:
            if timing_key in metrics:
                timing = metrics[timing_key]
                samples = compute_consumed_mllog_tokens(self.trainer, self.model)
                loss_data = {}
                if self.train_loss_key in metrics:
                    loss_data[self.train_loss_key] = metrics[self.train_loss_key]
                if os.environ.get("USE_DATETIME", "0") == "1":
                    mllogger.event(key='tracked_stats', metadata={'step': samples},
                        value={timing_key: timing, **loss_data, 'time_now': str(datetime.now())})
                else:
                    mllogger.event(key='tracked_stats', metadata={'step': samples},
                        value={timing_key: timing, **loss_data})

        if self.throughput_key in metrics:
            timing = metrics[self.throughput_key]
            samples = compute_consumed_mllog_tokens(self.trainer, self.model)
            throughput = samples / timing
            mllogger.event(key='tracked_stats', metadata={'step': samples},
                           value={'throughput': throughput})

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace],
                        *args: Any, **kwargs: Any) -> None:
        model_cfg = params.cfg
        mllogger.mlperf_submission_log('gpt3')

        mllogger.event(key=mllogger.constants.SEED, value=model_cfg.seed,
                       sync=False, unique=True)
        mllogger.event(key=mllogger.constants.GLOBAL_BATCH_SIZE,
                       value=model_cfg.global_batch_size, sync=False)
        b1, b2 = model_cfg.optim.betas
        mllogger.event(key="opt_name", value="adam", sync=False, unique=True)
        mllogger.event(key=mllogger.constants.OPT_BASE_LR,
                       value=model_cfg.optim.lr, sync=False, unique=True)
        mllogger.event(key="opt_end_learning_rate",
                       value=model_cfg.optim.sched.min_lr, sync=False, unique=True)
        mllogger.event(key="opt_adam_beta_1", value=b1, sync=False, unique=True)
        mllogger.event(key="opt_adam_beta_2", value=b2, sync=False, unique=True)
        mllogger.event(key="opt_adam_epsilon",
                       value=self.model.optimizers().optimizer.param_groups[0]['eps'], sync=False, unique=True)
        mllogger.event(key="opt_weight_decay",
                       value=model_cfg.optim.weight_decay, sync=False, unique=True)
        mllogger.event(key="opt_learning_rate_decay_steps",
                       value=int(model_cfg.optim.sched.max_steps_for_lr_sched), sync=False, unique=True)
        mllogger.event(key="opt_learning_rate_warmup_steps",
                       value=int(model_cfg.optim.sched.warmup_steps), sync=False, unique=True)
        mllogger.event(key="opt_learning_rate_decay_schedule",
                       value="cosine with linear warmup", sync=False, unique=True)
        mllogger.event(key="opt_gradient_clip_norm",
                       value=self.trainer.gradient_clip_val, sync=False, unique=True)
        mllogger.event(key="init_checkpoint_step",
                       value=model_cfg.custom.init_global_step, sync=False, unique=True)
        mllogger.event(key=mllogger.constants.GRADIENT_ACCUMULATION_STEPS,
                       value=get_num_microbatches(), sync=False, unique=True)
        mllogger.event(key="max_sequence_length",
                       value=model_cfg.encoder_seq_length, sync=False, unique=True)
        mllogger.event(key=mllogger.constants.EVAL_SAMPLES,
                       value=11590004, sync=False, unique=True)

    @property
    def name(self) -> Optional[str]:
        return 'mlperf-metrics'

    @property
    def version(self) -> Optional[Union[int, str]]:
        return 1

class EpochTimingCallback(TimingCallback):
    def __init__(self, timer: NamedTimer):
        # NOTE: don't call super().__init__() to reuse timer
        self.timer = timer

    def _on_epoch_start(self, name):
        self._on_batch_start(name)

    def _on_epoch_end(self, name, pl_module):
        self.timer.stop(name)
        pl_module.log(name, self.timer[name], on_step=False, on_epoch=True, batch_size=1)

    def on_validation_epoch_start(self, trainer, pl_module):
        self._on_batch_start("validation_epoch_timing")

    def on_validation_epoch_end(self, trainer, pl_module):
        self._on_epoch_end("validation_epoch_timing", pl_module)

    def on_train_epoch_start(self, trainer, pl_module):
        self._on_batch_start("train_epoch_timing")

    def on_train_epoch_end(self, trainer, pl_module):
        self._on_epoch_end("train_epoch_timing", pl_module)

def print_once(*args, **kwargs):
    if torch.distributed.get_rank():
        return
    print(*args, **kwargs)

class DeltaTimingCallback(Callback):
    def __init__(self):
        self.t0 = None
        self.total_train_step_time = [0, 0] # total_time, num of iterations
        self.total_valid_step_time = [0, 0]

    def trigger_tick(self):
        self.t0 = time.time()

    def on_train_start(self, trainer, pl_module):
        self.t0 = time.time()
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        t1 = time.time()
        d = t1 - self.t0
        self.total_train_step_time[0] += d
        print_once(f'Step {self.total_train_step_time[1]} train_step_timing {d} timestamp: {t1*1000}')
        self.total_train_step_time[1] += 1
        self.t0 = t1

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        t1 = time.time()
        d = t1 - self.t0
        self.total_valid_step_time[0] += d
        self.total_valid_step_time[1] += 1
        print_once(f'validation_step_timing {d} timestamp: {t1*1000}')
        self.t0 = t1

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print_once(f'Average train_step_time {self.total_train_step_time[0]/self.total_train_step_time[1]}\n \
Average valid_step_time {self.total_valid_step_time[0]/self.total_valid_step_time[1]} \
')

class CustomMegatronGPTModel(MegatronGPTModel):
    # setup_training_data, on_train_start and setup_data_mmap are here to move dataloader initialization past RUN_START tag
    # PTL has iter(dl) call in it's fit_loop.setup_data method, which spawns 2 processes that immediately prefetch data 
    def setup_training_data(self, cfg):
        if not self.cfg.data.delay_data_init:
            return super().setup_training_data(self.cfg.data)

        if hasattr(self, '_train_ds'):
            consumed_samples = self.compute_consumed_samples(0)
            # Assign a dummy dataloader with MockGPTDataset for self._train_dl to run PTL's setup_data() method so that the actual data is not prefetched
            # during the iter() call in setup_data().
            self._train_dl = self.build_pretraining_data_loader(
                MockGPTDataset(
                    cfg,
                    self.tokenizer,
                    "train",
                    num_samples = self.cfg.global_batch_size * self.trainer.max_steps,
                    seq_length = cfg.seq_length,
                    seed = 1
                ),
                consumed_samples,
            )

    def on_train_start(self) -> None:
        # Call on_train_start of MegatronBaseModel
        super().on_train_start()
        if not self.cfg.data.delay_data_init:
            return

        # Build the actual dataloader with self._train_ds
        consumed_samples = self.compute_consumed_samples(0)
        self._train_dl = self.build_pretraining_data_loader(self._train_ds, consumed_samples)
        # Setyp MMap before fit_loop initializes dataloader but after the run_start tag
        self.setup_data_mmap()
        # Reset fit_loop._combined_loader to None.
        self.trainer.fit_loop._combined_loader = None
        # Redo setup_data from PTL's fit_loop.py
        self.trainer.fit_loop.setup_data()

    def setup_data_mmap(self):
        if self.cfg.data.get('delay_data_mmap', False) and not isinstance(self._train_ds, MockGPTDataset):
            if self._train_ds:
                self._train_ds.create_data_mmap()
            if self._validation_ds:
                self._validation_ds.create_data_mmap()
            if self._test_ds:
                self._test_ds.create_data_mmap()

    def get_synthetic_input_training(self):
        if isinstance(self._train_ds, MockGPTDataset):
            single_data = self._train_ds[0]
        else:
            text = torch.ones(self.cfg.data.seq_length + 1, dtype=torch.int64) * 3545  # some token
            text[-1] = 0

            tokens = text[:-1].contiguous()
            labels = text[1:].contiguous()

            train_ds = self._train_ds.datasets[0]

            if self.cfg.data.legacy_dataset:
                arg_list = [tokens, train_ds.eos_id, train_ds.reset_position_ids,
                            train_ds.reset_attention_mask, train_ds.eod_mask_loss, False]
            else:
                arg_list = [tokens, train_ds.config.tokenizer.eod, train_ds.config.reset_position_ids,
                            train_ds.config.reset_attention_mask, train_ds.config.eod_mask_loss, False]
            attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(*arg_list)

            single_data = {
                'tokens': tokens,
                'labels': labels,
                'loss_mask': loss_mask,
                'position_ids': position_ids,
            }
        if isinstance(self._train_dl.batch_sampler, BaseMegatronBatchSampler):
            batch = default_collate([single_data] * self.cfg.micro_batch_size * get_num_microbatches())
        elif isinstance(self._train_dl.batch_sampler, MegatronPretrainingSampler):
            batch = default_collate([single_data] * self.cfg.micro_batch_size)
            batch = repeat(batch)
        else:
            raise NotImplementedError(f'No synthetic data implementation for data sampler "{self._train_dl.batch_sampler}"')
        return batch

    def get_synthetic_input_validation(self):
        seq_length = self.cfg.data.seq_length
        text = torch.ones(seq_length + 1, dtype=torch.int64) * 3545  # some token
        text[-1] = 0

        tokens = text[:-1].contiguous()
        labels = text[1:].contiguous()

        loss_mask = torch.ones(seq_length, dtype=torch.float32)
        loss_mask[-1] = 0.

        position_ids = torch.tensor([i for i in range(seq_length)], dtype=torch.int64)

        single_data = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
        }
        if isinstance(self._validation_dl.batch_sampler, BaseMegatronBatchSampler):
            batch = default_collate([single_data] * self.cfg.micro_batch_size * get_num_microbatches())
        elif isinstance(self._validation_dl.batch_sampler, MegatronPretrainingSampler):
            batch = default_collate([single_data] * self.cfg.micro_batch_size)
            batch = repeat(batch)
        else:
            raise NotImplementedError(f'No synthetic data implementation for data sampler "{self._validation_dl.batch_sampler}"')
        return batch

    def _register_sharded_tensor_state_dict_hooks_if_available(self) -> None:
        logger.info('Overriding _register_sharded_tensor_state_dict_hooks_if_available'
                     ' to mitigate incompatibility of PTL and PyTorch')
        return

    def _extract_consumed_samples_from_ckpt(self, ckpt_path):
        consumed_samples = super()._extract_consumed_samples_from_ckpt(ckpt_path)
        if consumed_samples == 0 and self.cfg.custom.override_zero_consumed_samples:
            consumed_samples = self.cfg.custom.init_global_step * self.cfg.global_batch_size
            logger.info(f'Overriding consumed_samples from 0 to {consumed_samples}')
        return consumed_samples

    def set_training(self, value):
        self.training = value

def configure_pre_validation_training_loop(trainer: pytorch_lightning.Trainer) -> None:
    if type(trainer.fit_loop.epoch_loop) != TrainingEpochLoop and not isinstance(trainer.fit_loop.epoch_loop, SkipResumeTrainingValidationLoop):
        return
    loop = PreValidationTrainingValidationLoop(trainer.min_steps, trainer.max_steps)
    loop.trainer = trainer
    trainer.fit_loop.epoch_loop = loop


class PreValidationTrainingValidationLoop(TrainingEpochLoop):
    """
    Extend the PTL Epoch loop to run validating on start.
    """

    def __init__(self, min_steps: Optional[int] = None, max_steps: int = -1) -> None:
        super().__init__(min_steps, max_steps)
        self.restarting = True

    def _should_check_val_fx(self) -> bool:
        if self.restarting and self.global_step == 0:
            return True
        return super()._should_check_val_fx()


def setup_auxiliary_loggers(log_marker='AUX'):
    """ Sets up non-NeMo loggers. Must be called after NeMo logging is set up.

    - Adds formatting to all logs
    - Removes INFO handlers on non-zero-ranks
    """
    class CustomFormatter(BaseNeMoFormatter):
        DEFAULT_FORMAT = BaseNeMoFormatter.DEFAULT_FORMAT.replace('NeMo', log_marker)

    class CustomDebugFormatter(DebugNeMoFormatter):
        DEFAULT_FORMAT = DebugNeMoFormatter.DEFAULT_FORMAT.replace('NeMo', log_marker)

    root = logging.getLogger()
    if not root.handlers:
        logger.warning(f'Failed to setup auxiliary loggers. Empty root logger handlers')
        return

    root_handler = root.handlers[0]
    if not isinstance(root_handler, logging.StreamHandler):
        logger.warning(f'Failed to setup auxiliary loggers. Unexpected root logger handler: {root.handlers[0]}')
        return

    if get_envbool(NEMO_ENV_VARNAME_TESTING, False):
        root_handler.setFormatter(CustomDebugFormatter())
        root.setLevel(logging.DEBUG)
    elif is_global_rank_zero():
        root_handler.setFormatter(CustomFormatter())
    else:
        # removing INFO handlers for non-zero ranks
        root.handlers.clear()
