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
# Based on: 
# https://github.com/wuphilipp/gello_software/blob/main/gello/dynamixel/driver.py
# ---------------------------------------------------------------------------


from threading import Lock
from typing import Protocol, Sequence

import numpy as np
from dynamixel_sdk.group_sync_read import GroupSyncRead
from dynamixel_sdk.group_sync_write import GroupSyncWrite
from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.port_handler import PortHandler
from dynamixel_sdk.robotis_def import (
    COMM_SUCCESS,
    DXL_HIBYTE,
    DXL_HIWORD,
    DXL_LOBYTE,
    DXL_LOWORD,
)

ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_CURRENT = 102
LEN_GOAL_CURRENT = 2
ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4
ADDR_PRESENT_VELOCITY = 128
LEN_PRESENT_VELOCITY = 4
ADDR_GOAL_POSITION = 116
LEN_GOAL_POSITION = 4
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
ADDR_OPERATING_MODE = 11
CURRENT_CONTROL_MODE = 0
POSITION_CONTROL_MODE = 3

TORQUE_TO_CURRENT_MAPPING = {
    "XC330_T288_T": 1158.73,
    "XM430_W210_T": 1000/2.69,
}


class DynamixelDriverProtocol(Protocol):
    def set_current(self, currents: Sequence[float]):
        """Set the current for the Dynamixel servos.

        Args:
            currents (Sequence[float]): A list of currents in mA.
        """
        ...

    def torque_enabled(self) -> bool:
        """Check if torque is enabled for the Dynamixel servos.

        Returns:
            bool: True if torque is enabled, False if it is disabled.
        """
        ...

    def set_torque_mode(self, enable: bool):
        """Set the torque mode for the Dynamixel servos.

        Args:
            enable (bool): True to enable torque, False to disable.
        """
        ...

    def get_positions(self) -> np.ndarray:
        """Get the current joint angles in radians.

        Returns:
            np.ndarray: An array of joint angles.
        """
        ...

    def close(self):
        """Close the driver."""


class DynamixelDriver(DynamixelDriverProtocol):
    def __init__(self, ids: Sequence[int], servo_types: Sequence[str], port: str = "/dev/ttyUSB0", baudrate: int = 4000000):
        self._ids = ids
        self._positions = None
        self._lock = Lock()

        self._portHandler = PortHandler(port)
        self._packetHandler = PacketHandler(2.0)
        self._groupSyncRead = GroupSyncRead(
            self._portHandler, self._packetHandler, ADDR_PRESENT_VELOCITY, LEN_PRESENT_POSITION + LEN_PRESENT_VELOCITY,
        )
        self._groupSyncWrite = GroupSyncWrite(
            self._portHandler, self._packetHandler, ADDR_GOAL_CURRENT, LEN_GOAL_CURRENT,
        )
        if not self._portHandler.openPort():
            raise RuntimeError("Failed to open the port")
        if not self._portHandler.setBaudRate(baudrate):
            raise RuntimeError(f"Failed to change the baudrate, {baudrate}")

        for dxl_id in self._ids:
            if not self._groupSyncRead.addParam(dxl_id):
                raise RuntimeError(f"Failed to add parameter for Dynamixel with ID {dxl_id}")
        
        self.torque_to_current_map = np.array(
            [TORQUE_TO_CURRENT_MAPPING[servo] for servo in servo_types]
        )

        self._torque_enabled = False
        try:
            self.set_torque_mode(self._torque_enabled)
        except Exception as e:
            print(f"port: {port}, {e}")

    @property
    def torque_enabled(self) -> bool:
        return self._torque_enabled

    def set_torque_mode(self, enable: bool):
        torque_value = TORQUE_ENABLE if enable else TORQUE_DISABLE
        with self._lock:
            for dxl_id in self._ids:
                dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(
                    self._portHandler, dxl_id, ADDR_TORQUE_ENABLE, torque_value
                )
                if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                    raise RuntimeError(f"Failed to set torque mode for Dynamixel with ID {dxl_id}")
        self._torque_enabled = enable

    def close(self):
        self._portHandler.closePort()

    def set_operating_mode(self, mode: int):
        for dxl_id in self._ids:
            dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(
                self._portHandler, dxl_id, ADDR_OPERATING_MODE, mode
            )
            if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                raise RuntimeError(f"Failed to set operating mode for Dynamixel with ID {dxl_id}")
    
    def verify_operating_mode(self, expected_mode: int):
        for dxl_id in self._ids:
            mode, dxl_comm_result, dxl_error = self._packetHandler.read1ByteTxRx(
                self._portHandler, dxl_id, ADDR_OPERATING_MODE
            )
            if dxl_comm_result != COMM_SUCCESS or dxl_error != 0 or mode != expected_mode:
                raise RuntimeError(f"Operating mode mismatch for Dynamixel ID {dxl_id}")

    def get_positions_and_velocities(self, tries=10):
        _positions = np.zeros(len(self._ids), dtype=int)
        _velocities = np.zeros(len(self._ids), dtype=int)
        
        # perform the group sync read transaction
        dxl_comm_result = self._groupSyncRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            if tries > 0:
                return self.get_positions_and_velocities(tries-1)
            else:
                raise RuntimeError(f"Warning, communication failed: {dxl_comm_result}")
        
        for i, dxl_id in enumerate(self._ids):
            # read velocity data
            if self._groupSyncRead.isAvailable(dxl_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY):
                velocity = self._groupSyncRead.getData(dxl_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)
                # apply sign correction
                if velocity > 0x7FFFFFFF:
                    velocity -= 0x100000000
                _velocities[i] = velocity
            else:
                raise RuntimeError(f"Failed to get velocity for Dynamixel with ID {dxl_id}")
            
            # read position data
            if self._groupSyncRead.isAvailable(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                position = self._groupSyncRead.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                # apply sign correction
                if position > 0x7FFFFFFF:
                    position -= 0x100000000
                _positions[i] = position
            else:
                raise RuntimeError(f"Failed to get position for Dynamixel with ID {dxl_id}")
            
        self._positions = _positions
        self._velocities = _velocities
        
        # return positions and velocities in meaningful units
        positions_in_radians = _positions / 2048.0 * np.pi
        velocities_in_units = _velocities * 0.229 * 2 * np.pi / 60
        
        return positions_in_radians, velocities_in_units
    

    def set_current(self, currents: Sequence[float]):
        if len(currents) != len(self._ids):
            raise ValueError("The length of currents must match the number of servos")
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set currents")

        currents = np.clip(currents, -900, 900)
        for dxl_id, current in zip(self._ids, currents):
            current_value = int(current)

            param_goal_current = [
                DXL_LOBYTE(current_value),
                DXL_HIBYTE(current_value)
            ]

            if not self._groupSyncWrite.addParam(dxl_id, param_goal_current):
                raise RuntimeError(f"Failed to set current for Dynamixel with ID {dxl_id}")
        dxl_comm_result = self._groupSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            raise RuntimeError("Failed to syncwrite goal current")
        self._groupSyncWrite.clearParam()

    def set_torque(self, torques: Sequence[float]):
        currents = self.torque_to_current_map*torques
        self.set_current(currents)
     

def main():
    # script for testing purposes
    ids = [1, 2, 3, 4, 5, 6, 7]
    port = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT8ISV6J-if00-port0"

    try:
        driver = DynamixelDriver(ids, port=port, baudrate=4000000)
    except FileNotFoundError:
        print(f"Port {port} not found. Please check the connection.")
        return
    
    driver.set_operating_mode(0)
    driver.set_torque_mode(True)

    try:
        while True:
            positions = driver.get_positions()
            print(f"Current joint positions for IDs {ids}: {positions}")

            current_values = [0, 0, 0, 0, 0, 0, 0.0]
            driver.set_current(current_values)
    except KeyboardInterrupt:
        driver.set_torque_mode(False)
        driver.close()

if __name__ == "__main__":
    main()
