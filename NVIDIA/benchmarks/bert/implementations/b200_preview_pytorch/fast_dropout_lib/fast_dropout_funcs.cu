/* Copyright (c) 2019-2024 NVIDIA CORPORATION. All rights reserved.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#include <vector>
#include <iostream>

#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <torch/extension.h>
#include <assert.h>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
struct ull2 {
    unsigned long long x;
    unsigned long long y;
};


inline __device__ uint16_t packed_le_8bit(unsigned A, unsigned B) {
    unsigned T=A & 0x7F7F7F7F;
    unsigned D=B | 0x80808080;
    D = -T + D;
    T = (A ^ B) | 0x7F7F7F7F;
    D = ~(T ^ D);
    asm ("lop3.b32 %0, %1, %2, %3, 0x4d;\n\t" : "=r"(D) : "r"(A), "r"(B), "r"(D));
    D = D & 0x80808080;
    T = (D & 0x0000FFFF) | (D >> 17);
    return T;
}


__global__ void dropout_add_8bit_rng(void* in_data, void *residual, at::PhiloxCudaState philox_args, void *output, uint16_t *out_mask, const size_t n_rng_blocks, const uint32_t p_dropout_in_uint32_t, const float inv_prob) {
    int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    int num_threads = gridDim.y * gridDim.x * blockDim.x;
    uint16_t result = 0;
    auto seeds = at::cuda::philox::unpack(philox_args);
    size_t ofs = tid;
    curandStatePhilox4_32_10_t state;
    curand_init(std::get<0>(seeds),
              ofs,
              std::get<1>(seeds),
              &state);
    ull2 input8;
    ull2 residual8;
    ull2* gmem_input_ptr = reinterpret_cast<ull2 *>(in_data);
    ull2* gmem_residual_ptr = reinterpret_cast<ull2 *>(residual);
    ull2* gmem_output_ptr = reinterpret_cast<ull2 *>(output);
  
    while (tid < n_rng_blocks) {
        uint4 random_uint4 = curand4(&state);
        result = packed_le_8bit(random_uint4.x, p_dropout_in_uint32_t);
        result |= packed_le_8bit(random_uint4.y, p_dropout_in_uint32_t) >> 2;
        result |= packed_le_8bit(random_uint4.z, p_dropout_in_uint32_t) >> 4;
        result |= packed_le_8bit(random_uint4.w, p_dropout_in_uint32_t) >> 6;

        uint16_t result_copy = result;
        #pragma unroll
	for (int j=0; j<2; j++) {
            input8 = gmem_input_ptr[tid*2+j];
            half *input_h_8 = reinterpret_cast<half *>(&input8);

            residual8 = gmem_residual_ptr[tid*2+j];
            half *residual_h_8 = reinterpret_cast<half *>(&residual8);

            #pragma unroll
            for (int i=0; i<8; i++) {
                input_h_8[i] = (result_copy & 0x1 == 0x1) ? (half)0.f : input_h_8[i]*(half)inv_prob;
		input_h_8[i] += residual_h_8[i];
	        result_copy = result_copy >> 1;
            }
            gmem_output_ptr[tid*2+j] = input8;
	}

        out_mask[tid] = result;//0xFFFF
        tid += num_threads;
        ofs = tid;
        result = 0;
    }
}


__global__ void dropout_add_8bit_rng_bwd(void* grad_out, uint16_t *in_mask, void *grad_in, const size_t n_rng_blocks, const float inv_prob) {
    int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    int num_threads = gridDim.y * gridDim.x * blockDim.x;
    uint16_t result = 0;
    ull2 input8;
    ull2* gmem_input_ptr = reinterpret_cast<ull2 *>(grad_out);
    ull2* gmem_output_ptr = reinterpret_cast<ull2 *>(grad_in);
  
    while (tid < n_rng_blocks) {
        result = in_mask[tid];
        #pragma unroll
	for (int j=0; j<2; j++) {
            input8 = gmem_input_ptr[tid*2+j];
            half *input_h_8 = reinterpret_cast<half *>(&input8);

            #pragma unroll
            for (int i=0; i<8; i++) {
                input_h_8[i] = (result & 0x1 == 0x1) ? (half)0.f : input_h_8[i]*(half)inv_prob;
	        result = result >> 1;
            }
	    gmem_output_ptr[tid*2+j] = input8;
	}

        tid += num_threads;
    }
}


std::vector<at::Tensor> dropout_add_fwd(at::Tensor &input,
               at::Tensor &residual,
               const float p_dropout)
{
    const int UNROLL=16;
    size_t numel = input.numel();
    TORCH_CHECK (numel==residual.numel(), "input and residual should have same shape");
    TORCH_CHECK (numel % 16 == 0, "input shape should be multiple of 16");
    TORCH_CHECK (input.scalar_type()==at::ScalarType::Half, "input type should be float16");
    TORCH_CHECK (residual.scalar_type()==at::ScalarType::Half, "residual type should be float16");

    auto output = torch::empty_like(input);
    long mask_size = numel / 16;
    auto options = torch::TensorOptions().dtype(at::ScalarType::UInt16).device(torch::kCUDA);
    auto mask_tensor = torch::empty({mask_size}, options);

    void *residual_ptr = static_cast<void*>(residual.data_ptr());
    void *input_ptr = static_cast<void*>(input.data_ptr());
    void *output_ptr = static_cast<void*>(output.data_ptr());
    void *mask_ptr = static_cast<void*>(mask_tensor.data_ptr());
    int blk_size = 128;
    int n_rng_blks = numel/16;
    int grid = (n_rng_blks+blk_size-1)/blk_size;
    dim3 dim_grid(grid);
    dim3 dim_block(blk_size);

    uint8_t p_dropout_in_uint8_t = static_cast<uint8_t>(std::floor(p_dropout * 255.0));
    unsigned p_dropout_in_uint32_t_8bit = 0;
    unsigned p_drop8 = p_dropout_in_uint8_t;
    for (int i=0; i<4; i++) {
        p_dropout_in_uint32_t_8bit = p_dropout_in_uint32_t_8bit | p_drop8;
        p_drop8 = p_drop8 << 8;
    }
    assert (p_dropout != 1.0f);
    float inv_prob = 1/(1-p_dropout);

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
		    std::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
    int64_t counter_offset = ((numel - 1)/(blk_size*grid*UNROLL)+1)*UNROLL;
    at::PhiloxCudaState rng_engine_inputs;
    {
        std::lock_guard<std::mutex> lock(gen->mutex_);
        rng_engine_inputs = gen->philox_cuda_state(counter_offset);
    }
    dropout_add_8bit_rng<<<dim_grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
        input_ptr,
        residual_ptr,
        rng_engine_inputs,
        output_ptr,
        reinterpret_cast<uint16_t *>(mask_ptr), mask_size, p_dropout_in_uint32_t_8bit, inv_prob);

    return {output, mask_tensor};
}


at::Tensor dropout_bwd(at::Tensor &gradout,
               at::Tensor &mask_tensor,
               const float p_dropout)
{
    size_t numel = gradout.numel();
    TORCH_CHECK (numel % 16 == 0, "input shape should be multiple of 16");
    TORCH_CHECK (gradout.scalar_type()==at::ScalarType::Half, "gradout type should be float16");

    auto gradin = torch::empty_like(gradout);
    long mask_size = numel / 16;

    void *gradout_ptr = static_cast<void*>(gradout.data_ptr());
    void *gradin_ptr = static_cast<void*>(gradin.data_ptr());
    void *mask_ptr = static_cast<void*>(mask_tensor.data_ptr());
    int blk_size = 128;
    int n_rng_blks = numel/16;
    int grid = (n_rng_blks+blk_size-1)/blk_size;
    dim3 dim_grid(grid);
    dim3 dim_block(blk_size);

    assert (p_dropout != 1.0f);
    float inv_prob = 1/(1-p_dropout);

    dropout_add_8bit_rng_bwd<<<dim_grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
        gradout_ptr,
        reinterpret_cast<uint16_t *>(mask_ptr),
        gradin_ptr,
        mask_size, inv_prob);

    return gradin;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dropout_add_fwd", &dropout_add_fwd, "dropout_add_fwd");
  m.def("dropout_bwd", &dropout_bwd, "dropout_bwd");
}
