# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/dgl/dgl:24.09-py3
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace/gnn
ENV WORK_DIR=/workspace/gnn

ARG GIT_COMMIT_ID
ENV GIT_COMMIT_ID=$GIT_COMMIT_ID

COPY . .
# required packages not included in DGL container: 
RUN pip install -r requirements.txt

## TransformerEngine
ARG TE_REVISION=SKIP
ENV CUSTOM_TE_REVISION ${TE_REVISION}

ENV WG_LOAD_USE_DIRECTIO=1

CMD /bin/bash
