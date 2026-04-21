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

# ---------------------------------------------------------------------------
# ZMQ address configuration
# ---------------------------------------------------------------------------
# If you need to communicate across machines (e.g. robot PC + data collection PC),
# update the addresses below to match your network configuration.
# Both processes must agree on which side binds (server) and which connects (client).
# By default, both DIRECT and the DROID plugin run on the same machine (localhost).
# ---------------------------------------------------------------------------

# ZMQ addresses for DIRECT leader arm ↔ DROID policy communication.
# Use localhost so both can run on the same machine. Ports can be changed if needed.

franka_direct_zmq_addresses = {
    "joint_state_sub": "tcp://127.0.0.1:7101",
    "joint_torque_sub": "tcp://127.0.0.1:7102",
    "joint_pos_cmd_pub": "tcp://127.0.0.1:7103",
}
