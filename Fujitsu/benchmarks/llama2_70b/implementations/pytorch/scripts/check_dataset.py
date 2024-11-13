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

import argparse
import subprocess

from hash import hash_directory

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="/data", type=str, help="Path to the model location")
args = parser.parse_args()

directory_hash = hash_directory(args.data_dir)
assert directory_hash == "682a5f40b790a56751bf8303554efc08", f"Expected hash 682a5f40b790a56751bf8303554efc08, but got {directory_hash}"
print(f"Succesfully downloaded and verified dataset with hash {directory_hash}")
