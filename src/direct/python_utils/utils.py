# ---------------------------------------------------------------------------
# FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning
# https://arxiv.org/abs/2502.17432
# Copyright (c) 2025 Jason Jingzhou Liu and Yulong Li

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
# ---------------------------------------------------------------------------


import os
import sys


def get_workspace_root():
    workspace_installation_root = os.environ.get("COLCON_PREFIX_PATH", "").split(os.pathsep)[0]
    workspace_root = os.path.abspath(os.path.join(workspace_installation_root, ".."))
    return workspace_root


def add_external_path(path):
    workspace_root = get_workspace_root()
    sys.path.append(os.path.join(workspace_root, path))
