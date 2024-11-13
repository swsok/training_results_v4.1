# Copyright (c) 2019-2023, NVIDIA CORPORATION. All rights reserved.
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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:24.05-py3

FROM ${FROM_IMAGE_NAME}

# Install dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        bzip2 \
        cabextract \
        iputils-ping \
        pbzip2 \
        pv \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/bert
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Preprocessing
WORKDIR /workspace
RUN cd /workspace && git clone https://github.com/attardi/wikiextractor.git
RUN cd /workspace/wikiextractor && git checkout e4abb4cbd019b0257824ee47c23dd163919b731b

# Install BERT
ENV BERT_PREP_WORKING_DIR /workspace/bert/data
WORKDIR /workspace/bert

# Only copy the mhalib here to avoid changing timestamp and rebuild TE (which takes 1.5 hours)
COPY ./mhalib ./mhalib

ENV PYTHONPATH "/workspace/bert"

RUN cd /workspace/bert/mhalib && python setup.py build && cp build/lib*/mhalib* ../

# Install TE
RUN MAX_JOBS=1 pip install git+https://github.com/NVIDIA/TransformerEngine.git@b8ba734

WORKDIR /workspace/bert

# pin scipy version
# otherwise https://github.com/scipy/scipy/issues/20813
RUN pip install scipy==1.11.4

# Update code
# Dummy argument to bust the cache
ARG CACHEBUST=1
RUN echo "Cache busting argument: $CACHEBUST"

COPY . .

# Increase open file limits
RUN echo "* soft nofile 65536" >> /etc/security/limits.conf && \
    echo "* hard nofile 65536" >> /etc/security/limits.conf && \
    echo "session required pam_limits.so" >> /etc/pam.d/common-session && \
    echo "session required pam_limits.so" >> /etc/pam.d/common-session-noninteractive

# Set ulimit for the current session
CMD ["bash", "-c", "ulimit -n 65536 && exec bash"]

