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

import threading

import numpy as np
import zmq


class ZMQSubscriber:
    """
    Simple ZMQ SUB/XSUB wrapper. Runs a background thread and exposes the last-received
    message as a numpy.ndarray (dtype=float64) via .message property.

    Args:
        ip_address: ZMQ address (e.g., "tcp://127.0.0.1:7101")
        verbose: Print debug messages
        bind_mode: If True, use XSUB and bind (server that receives from connecting PUBs).
                   If False, use SUB and connect (client that connects to a bound PUB).
    """

    def __init__(self, ip_address="tcp://127.0.0.1:7101", verbose=False, bind_mode=False):
        context = zmq.Context.instance()

        if bind_mode:
            # Server mode: use XSUB which can bind and receive from connecting PUBs
            self._sub_socket = context.socket(zmq.XSUB)
            self._sub_socket.bind(ip_address)
            # XSUB requires sending a subscription message (0x01 = subscribe, empty = all)
            self._sub_socket.send(b"\x01")
        else:
            # Client mode: use regular SUB and connect to a bound PUB
            self._sub_socket = context.socket(zmq.SUB)
            self._sub_socket.connect(ip_address)
            self._sub_socket.setsockopt(zmq.SUBSCRIBE, b"")

        # Use CONFLATE to always get the latest message, drop older ones if we're behind
        self._sub_socket.setsockopt(zmq.CONFLATE, True)
        # Set receive timeout to detect disconnects
        self._sub_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout for faster reconnect detection

        self._value = None
        self.verbose = verbose
        self.bind_mode = bind_mode
        self.ip_address = ip_address
        self.last_message = None
        self._running = True
        self._connected = False  # Track connection state
        self._subscriber_thread = threading.Thread(target=self._update_value, daemon=True)
        self._subscriber_thread.start()

    @property
    def message(self):
        if self._value is None and self.verbose:
            print("The subscriber has not received a message")
        self.last_message = self._value
        return self._value

    @property
    def is_connected(self):
        """Returns True if we've received at least one message recently."""
        return self._connected

    def _update_value(self):
        while self._running:
            try:
                message = self._sub_socket.recv()
                # interpret bytes as float64 array (producer uses float64)
                try:
                    arr = np.frombuffer(message, dtype=np.float64)
                except Exception:
                    # fallback: try float32 then cast
                    arr = np.frombuffer(message, dtype=np.float32).astype(np.float64)
                self._value = arr
                self._connected = True
            except zmq.Again:
                # Timeout - no message received, but keep trying
                # Mark as disconnected if we timeout
                self._connected = False
                continue
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM:
                    break  # Context terminated
                # Other errors - keep trying
                self._connected = False
                continue
            except Exception:
                # break only on explicit close
                break

    def close(self):
        self._running = False
        try:
            self._sub_socket.close(linger=0)
        except Exception:
            pass


class ZMQPublisher:
    """
    Simple ZMQ PUB wrapper. Sends numpy arrays as raw float64 bytes.

    Args:
        ip_address: ZMQ address (e.g., "tcp://127.0.0.1:7101")
        bind_mode: If True, bind (act as server). If False, connect (act as client).
    """

    def __init__(self, ip_address="tcp://127.0.0.1:7101", bind_mode=True):
        context = zmq.Context.instance()
        self._pub_socket = context.socket(zmq.PUB)
        # Set High Water Mark to buffer messages if subscribers are slow to connect
        self._pub_socket.setsockopt(zmq.SNDHWM, 1000)

        if bind_mode:
            # Server mode: bind and wait for subscribers to connect
            self._pub_socket.bind(ip_address)
        else:
            # Client mode: connect to a bound subscriber (XSUB pattern)
            self._pub_socket.connect(ip_address)

        self.bind_mode = bind_mode
        self.ip_address = ip_address
        self.last_message = None
        # Add a small delay to allow connections to establish
        import time

        time.sleep(0.1)

    def send_message(self, message):
        # accepts numpy array, list or scalar; always send float64 bytes
        arr = np.asarray(message, dtype=np.float64)
        self.last_message = arr
        # use tobytes for raw transport
        self._pub_socket.send(arr.tobytes())

    def close(self):
        try:
            self._pub_socket.close(linger=0)
        except Exception:
            pass
