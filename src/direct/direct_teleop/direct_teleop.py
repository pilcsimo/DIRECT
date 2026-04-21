import logging
import os
import subprocess
import sys
import threading
import time
from enum import IntEnum
from pathlib import Path
from typing import Optional

import numpy as np
import pinocchio as pin
import yaml

from direct.direct_teleop.dynamixel.driver import DynamixelDriver
from direct.python_utils.global_configs import franka_direct_zmq_addresses
from direct.python_utils.zmq_messenger import ZMQPublisher, ZMQSubscriber


class TeleopCommand(IntEnum):
    """Command codes sent from policy to teleop controller as 9th element."""

    NONE = 0
    HOME = 1
    MIRROR = 2  # Sync leader arm to follower robot position
    SET_NULL_SPACE = 3
    DISABLE = 4  # Explicitly disable movement (ACTIVE -> IDLE)


class TeleopState(IntEnum):
    """State machine states for teleop controller."""

    IDLE = 0  # Not actively controlling robot (movement disabled)
    SYNCING = 1  # Leader arm is syncing to robot position
    ACTIVE = 2  # Actively controlling robot (movement enabled)


def find_ttyusb(port_name):
    """
    This function is used to locate the underlying ttyUSB device.
    """
    base_path = "/dev/serial/by-id/"
    full_path = os.path.join(base_path, port_name)
    if not os.path.exists(full_path):
        raise Exception(f"Port '{port_name}' does not exist in {base_path}.")
    try:
        resolved_path = os.readlink(full_path)
        actual_device = os.path.basename(resolved_path)
        if actual_device.startswith("ttyUSB"):
            return actual_device
        else:
            raise Exception(
                f"The port '{port_name}' does not correspond to a ttyUSB device. It links to {resolved_path}."
            )
    except Exception as e:
        raise Exception(f"Unable to resolve the symbolic link for '{port_name}'. {e}") from e


class DIRECTTeleop:
    """
    Plain-Python DIRECT teleop class (no ROS). Runs a high-frequency control loop in a background
    thread and sends joint commands via ZMQ at a reduced rate (50 Hz) to avoid flooding the robot.
    """

    def __init__(
        self,
        config_file_name: str,
        address_mode: str = "droid",
        force_recalibrate: bool = False,
        calibration_file: Optional[str] = None,
    ):
        # load config
        direct_teleop_dir = os.path.dirname(__file__)
        config_path = os.path.join(direct_teleop_dir, "configs", config_file_name)
        with open(config_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)

        config_path_obj = Path(config_path)
        self._calibration_file_path = (
            Path(calibration_file) if calibration_file else config_path_obj.with_suffix(".calib.npy")
        )
        self._force_recalibrate = force_recalibrate

        # basic fields
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.frequency = float(self.config["controller"]["frequency"])
        self.dt = 1.0 / self.frequency

        # prepare hardware/network and models
        self._prepare_dynamixel()
        self._prepare_inverse_dynamics()

        # leader arm parameters
        arm_cfg = self.config["arm_teleop"]
        self.num_arm_joints = arm_cfg["num_arm_joints"]
        self.safety_margin = arm_cfg["arm_joint_limits_safety_margin"]
        self.arm_joint_limits_max = np.array(arm_cfg["arm_joint_limits_max"]) - self.safety_margin
        self.arm_joint_limits_min = np.array(arm_cfg["arm_joint_limits_min"]) + self.safety_margin
        init_cfg = arm_cfg.get("initialization", {})
        default_init = np.zeros(self.num_arm_joints)
        self.initial_match_joint_pos = np.array(init_cfg.get("initial_match_joint_pos", default_init))
        if self.initial_match_joint_pos.size != self.num_arm_joints:
            raise ValueError("initial_match_joint_pos length must match num_arm_joints")
        assert self.num_arm_joints == len(self.arm_joint_limits_max) == len(self.arm_joint_limits_min)

        # gripper
        self.gripper_limit_min = 0.0
        self.gripper_limit_max = self.config["gripper_teleop"]["actuation_range"]
        self.gripper_pos_prev = 0.0
        self.gripper_pos = 0.0

        # controller flags and gains
        self.enable_gravity_comp = self.config["controller"]["gravity_comp"]["enable"]
        self.gravity_comp_modifier = self.config["controller"]["gravity_comp"]["gain"]
        self.tau_g = np.zeros(self.num_arm_joints)

        self.stiction_comp_enable_speed = self.config["controller"]["static_friction_comp"]["enable_speed"]
        self.stiction_comp_gain = self.config["controller"]["static_friction_comp"]["gain"]
        self.stiction_dither_flag = np.ones((self.num_arm_joints), dtype=bool)

        self.joint_limit_kp = self.config["controller"]["joint_limit_barrier"]["kp"]
        self.joint_limit_kd = self.config["controller"]["joint_limit_barrier"]["kd"]

        self.null_space_joint_target = np.array(
            self.config["controller"]["null_space_regulation"]["null_space_joint_target"]
        )
        self.null_space_kp = self.config["controller"]["null_space_regulation"]["kp"]
        self.null_space_kd = self.config["controller"]["null_space_regulation"]["kd"]

        self.enable_torque_feedback = self.config["controller"]["torque_feedback"]["enable"]
        self.torque_feedback_gain = self.config["controller"]["torque_feedback"]["gain"]
        self.torque_feedback_motor_scalar = self.config["controller"]["torque_feedback"]["motor_scalar"]
        self.torque_feedback_damping = self.config["controller"]["torque_feedback"]["damping"]

        gripper_fb_cfg = self.config["controller"].get("gripper_feedback", {})
        self.enable_gripper_feedback = bool(gripper_fb_cfg.get("enable", False))
        self.gripper_feedback_gain = float(gripper_fb_cfg.get("gain", 1.0))
        self.gripper_feedback_damping = float(gripper_fb_cfg.get("damping", 0.1))

        # Per-joint gain multipliers (1.0 = use config values, <1.0 = reduce, >1.0 = increase)
        # Backward compatible: all default to 1.0 unless explicitly set
        self.per_joint_gain_multipliers = np.array([1.0, 1.25, 1.0, 1.5, 1.0, 1.0, 0.5])

        # Cache sync PD gains to avoid config dict lookups in the 500 Hz hot path
        _pc = self.config["controller"]["joint_position_control"]
        self._sync_kp = float(_pc["kp"])
        self._sync_kd = float(_pc["kd"])
        self._sync_interpolation_step = float(_pc["interpolation_step_size"])
        self._sync_kp_gains = self._sync_kp * self.per_joint_gain_multipliers
        self._sync_kd_gains = self._sync_kd * self.per_joint_gain_multipliers

        # communication (ZMQ) setup - DIRECT only needs to publish joint states
        # The address_mode parameter is kept for backward compatibility but no longer changes behavior
        self._address_mode = address_mode.lower()
        self.set_up_communication()

        # manual calibration state
        self.control_enabled = False
        # Initial offset: when teleop is at mechanical zero, robot should be at [0, 0, 0, 0, 0, 0, π]
        # Since robot_pos = (teleop_pos - offset) * sign, and we want robot_pos = π when teleop_pos = 0 for joint 6:
        # π = (0 - offset) * sign, so offset = -π * sign. With sign = 1 for joint 6: offset = -π
        self.joint_offsets = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -np.pi,
            0.0,
        ]  # Offsets to apply to leader arm joints (last joint offset by 180 degrees)
        self.gripper_pos_prev = 0.0
        self.gripper_pos = 0.0

        calibration_loaded = False
        if not self._force_recalibrate:
            calibration_loaded = self._load_calibration_from_file()

        if calibration_loaded:
            self.control_enabled = True
            self.logger.info(f"Loaded calibration from {self._calibration_file_path}")
        else:
            if self._force_recalibrate:
                self.logger.info("Forced manual calibration requested.")
            calibration_payload = self._perform_manual_calibration()
            self._save_calibration_to_file(calibration_payload)

        # control loop thread
        self._running = False
        self._thread = None
        # 50 Hz throttle for outgoing joint position commands
        self._cmd_publish_interval = 1.0 / 50.0
        self._last_cmd_publish_time = 0.0

        # State machine for teleop control
        self._teleop_state = TeleopState.IDLE
        self._target_robot_pos = None  # Target position to sync to
        self._target_robot_gripper = 0.0

        # All-joints sync tracking
        self._sync_pos_threshold = 0.1  # Radians - position error threshold for all joints
        self._sync_vel_threshold = 0.05  # Radians/sec - velocity threshold for all joints
        self._sync_stable_frames = 0  # Consecutive frames meeting position+velocity criteria
        self._sync_stable_threshold = 10  # Need 10 consecutive good frames (~20ms at 500Hz)

        # Patience-based per-joint kick for stuck joints during sync
        self._sync_error_stagnation_threshold = 200  # Frames of stagnation before kick triggers
        self._sync_error_improvement_threshold = 0.01  # Per-joint rad improvement to reset stagnation
        self._sync_kick_region = 0.5  # Per-joint error (rad) below which stagnation is tracked
        self._sync_kick_torque = 0.25  # Nm - discrete torque kick magnitude per kicked joint
        self._sync_kick_duration_frames = 25  # Frames each per-joint kick lasts
        # Per-joint tracking arrays (reset on each new sync via _reset_sync_patience)
        self._sync_joint_best_error = np.full(self.num_arm_joints, np.inf)
        self._sync_joint_stagnation_frames = np.zeros(self.num_arm_joints, dtype=int)
        self._sync_kick_frames = np.zeros(self.num_arm_joints, dtype=int)

        # Rich live sync display
        self._sync_live = None
        self._sync_heartbeat_frames = 0
        self._sync_heartbeat_interval = 50  # Update terminal display every 50 frames (~100 ms at 500 Hz)

        # Cached last leader position (avoids redundant hardware reads within one callback cycle)
        self._last_leader_pos = np.zeros(self.num_arm_joints)

        # Pre-allocated publish buffer to avoid np.append allocations in update_communication
        self._state_msg_buf = np.zeros(9)

        # Sync status signal: 0=IDLE, 1=SYNCING, 2=SYNC_COMPLETE (sent to policy)
        self._sync_status = 0.0

        # Hysteresis for drift detection
        self._sync_pos_drift_threshold = 0.3

    def _prepare_dynamixel(self):
        """
        Instantiates driver for interfacing with Dynamixel servos.
        """
        self.servo_types = self.config["dynamixel"]["servo_types"]
        self.num_motors = len(self.servo_types)
        self.joint_signs = np.array(self.config["dynamixel"]["joint_signs"], dtype=float)
        assert self.num_motors == len(self.joint_signs), (
            "The number of motors and the number of joint signs must be the same"
        )
        self.dynamixel_port = "/dev/serial/by-id/" + self.config["dynamixel"]["dynamixel_port"]

        # checks of the latency timer on ttyUSB of the corresponding port is 1
        # if it is not 1, the control loop cannot run at above 200 Hz, which will
        # cause extremely undesirable behaviour for the leader arm. If the latency
        # timer is not 1, one can set it to 1 as follows:
        # echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB{NUM}/latency_timer
        ttyUSBx = find_ttyusb(self.dynamixel_port)
        command = f"cat /sys/bus/usb-serial/devices/{ttyUSBx}/latency_timer"
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        ttyUSB_latency_timer = int(result.stdout)
        if ttyUSB_latency_timer != 1:
            raise Exception(
                f"Please ensure the latency timer of {ttyUSBx} is 1. Run: \n \
                echo 1 | sudo tee /sys/bus/usb-serial/devices/{ttyUSBx}/latency_timer"
            )

        joint_ids = np.arange(self.num_motors) + 1
        try:
            self.driver = DynamixelDriver(joint_ids, self.servo_types, self.dynamixel_port)
        except FileNotFoundError:
            self.logger.info(f"Port {self.dynamixel_port} not found. Please check the connection.")
            return
        self.driver.set_torque_mode(False)
        # set operating mode to current mode
        self.driver.set_operating_mode(0)
        # enable torque
        self.driver.set_torque_mode(True)

    def _prepare_inverse_dynamics(self):
        """
        Creates a model of the leader arm given the its URDF for kinematic and dynamic
        computations used in gravity compensation and null-space regulation calculations.
        """
        direct_teleop_dir = os.path.dirname(__file__)
        urdf_path = os.path.join(direct_teleop_dir, "urdf", self.config["arm_teleop"]["leader_urdf"])

        # buildModelFromUrdf is provided at the top-level pinocchio package
        # (use the correctly-cased API and call via the `pin` namespace)
        self.pin_model = pin.buildModelFromUrdf(urdf_path)
        self.pin_data = pin.Data(self.pin_model)

    def set_up_communication(self):
        """
        Set up ZMQ communication for teleoperation. DIRECT acts as the SERVER:
        - Binds a PUB socket to publish leader arm joint states
        - Binds SUB sockets to receive robot state and torque feedback from policy

        The policy (client) connects to these bound sockets.
        This allows DIRECT to start first and wait for the policy to connect.
        """
        self.zmq_addresses = franka_direct_zmq_addresses
        self.logger.info("Setting up ZMQ communication (SERVER mode)")

        # Publisher: send leader arm joint states to the policy (DIRECT binds)
        self.franka_cmd_pub = ZMQPublisher(self.zmq_addresses["joint_pos_cmd_pub"], bind_mode=True)
        self.logger.info(f"[SERVER] Publishing leader arm states on: {self.zmq_addresses['joint_pos_cmd_pub']}")

        # Subscriber: receive robot joint states from the policy (DIRECT binds, policy connects)
        self.franka_joint_state_sub = ZMQSubscriber(self.zmq_addresses["joint_state_sub"], bind_mode=True)
        self.logger.info(f"[SERVER] Listening for robot states on: {self.zmq_addresses['joint_state_sub']}")

        # Subscriber: receive external torque feedback if enabled (DIRECT binds)
        if self.enable_torque_feedback:
            self.franka_torque_sub = ZMQSubscriber(self.zmq_addresses["joint_torque_sub"], bind_mode=True)
            self.logger.info(f"[SERVER] Listening for torque feedback on: {self.zmq_addresses['joint_torque_sub']}")
        else:
            self.franka_torque_sub = None

        # Initialize gripper feedback
        self.gripper_external_torque = 0.0

        # Track policy connection state
        self._policy_connected = False
        self._last_policy_msg_time = 0.0

    def _load_calibration_from_file(self):
        if self._calibration_file_path is None:
            return False

        path = Path(self._calibration_file_path)
        if not path.exists():
            self.logger.info(f"No calibration file found at {path}, starting manual calibration.")
            return False

        try:
            stored = np.load(path, allow_pickle=True)
            data = stored.item() if isinstance(stored, np.ndarray) else stored
        except Exception as exc:
            self.logger.warning(f"Failed to load calibration from {path}: {exc}")
            return False

        if not isinstance(data, dict):
            self.logger.warning(f"Calibration file {path} is not a dictionary. Recalibrating.")
            return False

        offsets = np.asarray(data.get("joint_offsets"))
        if offsets is None or offsets.size != self.num_motors:
            self.logger.warning(f"Calibration data in {path} has invalid joint offsets. Recalibrating.")
            return False

        targets = data.get("null_space_joint_target", self.null_space_joint_target)
        targets = np.asarray(targets, dtype=float)
        if targets.size < self.num_arm_joints:
            self.logger.warning(f"Calibration data in {path} has incomplete null-space target. Recalibrating.")
            return False

        self.joint_offsets = offsets.astype(float)
        if targets.size != self.null_space_joint_target.size:
            updated = np.array(self.null_space_joint_target, copy=True)
            updated[: self.num_arm_joints] = targets[: self.num_arm_joints]
            targets = updated
        self.null_space_joint_target = targets.astype(float)

        self.gripper_pos_prev = 0.0
        self.gripper_pos = 0.0

        try:
            if getattr(self, "driver", None) is not None:
                self.driver.set_operating_mode(0)
                self.driver.set_torque_mode(True)
        except Exception as exc:
            self.logger.warning(f"Failed to ensure torque mode after loading calibration: {exc}")

        return True

    def _save_calibration_to_file(self, calibration_payload):
        if calibration_payload is None or self._calibration_file_path is None:
            return

        path = Path(self._calibration_file_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, calibration_payload, allow_pickle=True)
            self.logger.info(f"Saved calibration to {path}")
        except Exception as exc:
            self.logger.warning(f"Failed to save calibration to {path}: {exc}")

    def _perform_manual_calibration(self):
        """Simple two-step manual calibration confirmed via Enter key presses."""
        if getattr(self, "driver", None) is None:
            self.logger.warning("Dynamixel driver unavailable, skipping manual calibration.")
            self.control_enabled = True
            return None

        # Disable torque so the operator can freely move the leader arm.
        try:
            self.driver.set_torque_mode(False)
        except Exception as exc:
            self.logger.warning(f"Failed to disable torque before calibration: {exc}")

        self.logger.info("Manual calibration: bring the leader arm to mechanical zero with the gripper fully open.")
        try:
            input("Press Enter once the leader arm is at mechanical zero... ")
        except (EOFError, KeyboardInterrupt):
            self.logger.info("Calibration input skipped, using current pose as zero.")

        joint_pos, _ = self.driver.get_positions_and_velocities()
        self.joint_offsets = np.array(joint_pos, dtype=float)

        # Add desired offset for 7th joint so that teleop zero corresponds to robot [0,0,0,0,0,0,π,0]
        # Since robot_pos = (teleop_pos - offset) * sign, and we want robot_pos[6] = π when teleop_pos[6] = 0:
        # π = (0 - offset[6]) * sign[6], so offset[6] = -π * sign[6]
        desired_robot_offset = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -np.pi, 0.0])
        if len(self.joint_offsets) > 6:  # Ensure we have at least 7 joints
            self.joint_offsets[6] += -desired_robot_offset[6] * self.joint_signs[6]

        self.gripper_pos_prev = 0.0
        self.gripper_pos = 0.0

        self.logger.info("Move the leader arm to the desired null-space posture, close the trigger all the way.")
        try:
            input("Press Enter once the desired null-space posture is reached and the trigger is fully closed... ")
        except (EOFError, KeyboardInterrupt):
            self.logger.info("Null-space input skipped, retaining existing null-space target.")

        joint_pos, _ = self.driver.get_positions_and_velocities()
        leader_arm_zeroed = (
            np.array(joint_pos[0 : self.num_arm_joints], dtype=float) - self.joint_offsets[0 : self.num_arm_joints]
        ) * self.joint_signs[0 : self.num_arm_joints]

        if self.null_space_joint_target.size < self.num_arm_joints:
            raise ValueError("null_space_joint_target must include entries for all arm joints")
        updated_null_target = np.array(self.null_space_joint_target, copy=True)
        updated_null_target[: self.num_arm_joints] = leader_arm_zeroed
        self.null_space_joint_target = updated_null_target

        try:
            self.driver.set_operating_mode(0)
            self.driver.set_torque_mode(True)
        except Exception as exc:
            self.logger.warning(f"Failed to re-enable torque after calibration: {exc}")

        self.control_enabled = True
        self.logger.info("Calibration complete. Leader arm control enabled.")

        calibration_data = {
            "joint_offsets": self.joint_offsets.copy(),
            "null_space_joint_target": self.null_space_joint_target.copy(),
            "timestamp": time.time(),
        }
        return calibration_data

    def get_leader_joint_states(self):
        """
        Returns the current joint positions and velocities of the leader arm and gripper,
        aligned with the joint conventions (range and direction) of the follower arm.
        """
        self.gripper_pos_prev = self.gripper_pos
        joint_pos, joint_vel = self.driver.get_positions_and_velocities()
        joint_pos_arm = (
            joint_pos[0 : self.num_arm_joints] - self.joint_offsets[0 : self.num_arm_joints]
        ) * self.joint_signs[0 : self.num_arm_joints]
        self.gripper_pos = (joint_pos[-1] - self.joint_offsets[-1]) * self.joint_signs[-1]
        joint_vel_arm = joint_vel[0 : self.num_arm_joints] * self.joint_signs[0 : self.num_arm_joints]

        gripper_vel = (self.gripper_pos - self.gripper_pos_prev) / self.dt
        return joint_pos_arm, joint_vel_arm, self.gripper_pos, gripper_vel

    def set_leader_joint_pos(self, goal_joint_pos, goal_gripper_pos):
        """
        Moves the leader arm and gripper to a specified joint configuration using a PD control loop.
        This method is useful for aligning the leader arm with a desired configuration, such as
        matching the follower arm's configuration. It interpolates the motion toward the target
        position and applies torque commands based on a PD controller.

        **Note:** This function is not used by default in the main teleoperation loop. To ensure
        controller stability, please ensure the latency of Dynamixel servos is minimized such
        that the control loop frequency is at least 200 Hz. Otherwise, the PD controller tuning
        is unstable for low control frequencies.
        """
        interpolation_step_size = (
            np.ones(7) * self.config["controller"]["joint_position_control"]["interpolation_step_size"]
        )
        kp = float(self.config["controller"]["joint_position_control"]["kp"])
        kd = float(self.config["controller"]["joint_position_control"]["kd"])

        curr_pos, curr_vel, curr_gripper_pos, curr_gripper_vel = self.get_leader_joint_states()
        while np.linalg.norm(curr_pos - goal_joint_pos) > 0.1:
            next_joint_pos_target = np.where(
                np.abs(curr_pos - goal_joint_pos) > interpolation_step_size,
                curr_pos + interpolation_step_size * np.sign(goal_joint_pos - curr_pos),
                goal_joint_pos,
            )
            torque = -(kp * self.per_joint_gain_multipliers) * (curr_pos - next_joint_pos_target) - (
                kd * self.per_joint_gain_multipliers
            ) * (curr_vel)
            gripper_torque = -kp * (curr_gripper_pos - goal_gripper_pos) - kd * (curr_gripper_vel)
            self.set_leader_joint_torque(torque, gripper_torque)
            curr_pos, curr_vel, curr_gripper_pos, curr_gripper_vel = self.get_leader_joint_states()

    def set_leader_joint_torque(self, arm_torque, gripper_torque):
        """
        Applies torque to the leader arm and gripper.
        """
        arm_gripper_torque = np.append(arm_torque, gripper_torque)
        self.driver.set_torque(arm_gripper_torque * self.joint_signs)

    def joint_limit_barrier(self, arm_joint_pos, arm_joint_vel, gripper_joint_pos, gripper_joint_vel):
        """
        Computes joint limit repulsive torque to prevent the leader arm and gripper from
        exceeding the physical joint limits of the follower arm.

        This method implements a simplified control law compared to the one described in
        Section IX.B of the paper, while achieving the same protective effect. It applies
        repulsive torques proportional to the distance from the joint limits and the joint
        velocity when limits are approached or exceeded.
        """
        exceed_max_mask = arm_joint_pos > self.arm_joint_limits_max
        tau_l = (
            -self.joint_limit_kp * (arm_joint_pos - self.arm_joint_limits_max) - self.joint_limit_kd * arm_joint_vel
        ) * exceed_max_mask
        exceed_min_mask = arm_joint_pos < self.arm_joint_limits_min
        tau_l += (
            -self.joint_limit_kp * (arm_joint_pos - self.arm_joint_limits_min) - self.joint_limit_kd * arm_joint_vel
        ) * exceed_min_mask

        if gripper_joint_pos > self.gripper_limit_max:
            tau_l_gripper = (
                -self.joint_limit_kp * (gripper_joint_pos - self.gripper_limit_max)
                - self.joint_limit_kd * gripper_joint_vel
            )
        elif gripper_joint_pos < self.gripper_limit_min:
            tau_l_gripper = (
                -self.joint_limit_kp * (gripper_joint_pos - self.gripper_limit_min)
                - self.joint_limit_kd * gripper_joint_vel
            )
        else:
            tau_l_gripper = 0.0
        return tau_l, tau_l_gripper

    def gravity_compensation(self, arm_joint_pos, arm_joint_vel):
        """
        Computes joint torque for gravity compensation using inverse dynamics.
        This method uses the Recursive Newton-Euler Algorithm (RNEA), provided by the
        Pinocchio library, to calculate the torques required to counteract gravity
        at the current joint states. The result is scaled by a modifier to tune the
        compensation strength.

        This implementation corresponds to the gravity compensation strategy
        described in Section III.C of the paper.
        """
        self.tau_g = pin.rnea(self.pin_model, self.pin_data, arm_joint_pos, arm_joint_vel, np.zeros_like(arm_joint_vel))
        self.tau_g *= self.gravity_comp_modifier
        return self.tau_g

    def friction_compensation(self, arm_joint_vel):
        """
        Compute joint torques to compensate for static friction during teleoperation.

        This method implements static friction compensation as described in Equation 7,
        Section IX.A of the paper. It omits kinetic friction compensation, which was
        necessary in earlier hardware versions to achieve smooth teleoperation, but has
        since become unnecessary due to hardware improvements, such as weight reduction.
        """
        tau_ss = np.zeros(self.num_arm_joints)
        for i in range(self.num_arm_joints):
            if abs(arm_joint_vel[i]) < self.stiction_comp_enable_speed:
                if self.stiction_dither_flag[i]:
                    tau_ss[i] += self.stiction_comp_gain * abs(self.tau_g[i])
                else:
                    tau_ss[i] -= self.stiction_comp_gain * abs(self.tau_g[i])
                self.stiction_dither_flag[i] = ~self.stiction_dither_flag[i]
        return tau_ss

    def null_space_regulation(self, arm_joint_pos, arm_joint_vel):
        """
        Computes joint torques to perform null-space regulation for redundancy resolution
        of the leader arm.

        This method enables the specification of a desired null-space joint configuration
        via `self.null_space_joint_target`. It implements the control strategy described
        in Equation 3 of Section III.B in the paper, projecting a PD control law into
        the null space of the task Jacobian to achieve secondary objectives without
        affecting the primary task.
        """
        J = pin.computeJointJacobian(self.pin_model, self.pin_data, arm_joint_pos, self.num_arm_joints)
        J_dagger = np.linalg.pinv(J)
        null_space_projector = np.eye(self.num_arm_joints) - J_dagger @ J
        q_error = arm_joint_pos - self.null_space_joint_target[0 : self.num_arm_joints]
        tau_n = null_space_projector @ (-self.null_space_kp * q_error - self.null_space_kd * arm_joint_vel)
        return tau_n

    def set_redundancy_resolution(self, null_space_joint_target=None, kp=None, kd=None):
        """
        Update redundancy resolution parameters at runtime.

        Args:
            null_space_joint_target: Desired null-space joint configuration (array-like)
            kp: Proportional gain for null-space regulation
            kd: Derivative gain for null-space regulation
        """
        if null_space_joint_target is not None:
            self.null_space_joint_target = np.array(null_space_joint_target)
            self.logger.info(f"Updated null_space_joint_target: {self.null_space_joint_target}")
        if kp is not None:
            self.null_space_kp = kp
            self.logger.info(f"Updated null_space_kp: {self.null_space_kp}")
        if kd is not None:
            self.null_space_kd = kd
            self.logger.info(f"Updated null_space_kd: {self.null_space_kd}")

    def set_per_joint_gains(self, multipliers):
        """
        Set per-joint gain multipliers to scale kp and kd on a per-joint basis.

        Args:
            multipliers: Array-like of length num_arm_joints. Each value scales the
                        corresponding joint's gains (1.0 = no change, 0.5 = half gain, 2.0 = double gain).

        Example:
            teleop.set_per_joint_gains([1.0, 1.0, 0.8, 1.2, 1.0, 1.0, 1.0])
        """
        multipliers = np.asarray(multipliers, dtype=float)
        if multipliers.size != self.num_arm_joints:
            raise ValueError(f"Expected {self.num_arm_joints} multipliers, got {multipliers.size}")
        self.per_joint_gain_multipliers = multipliers.copy()
        self._sync_kp_gains = self._sync_kp * self.per_joint_gain_multipliers
        self._sync_kd_gains = self._sync_kd * self.per_joint_gain_multipliers
        self.logger.info(f"Updated per-joint gain multipliers: {np.round(self.per_joint_gain_multipliers, 3)}")

    def get_per_joint_gains(self):
        """Get current per-joint gain multipliers."""
        return self.per_joint_gain_multipliers.copy()

    def torque_feedback(self, external_torque, arm_joint_vel):
        """
        Computes joint torque for the leader arm to achieve force-feedback based on
        the external joint torque from the follower arm.

        This method implements Equation 1 in Section III.A of the paper.
        """
        tau_ff = -1.0 * self.torque_feedback_gain / self.torque_feedback_motor_scalar * external_torque
        tau_ff -= self.torque_feedback_damping * arm_joint_vel
        return tau_ff

    def control_loop_callback(self, now: Optional[float] = None):
        """
        Runs the main control loop of the leader arm.

        Note that while the control loop can run at up to 500 Hz, lower frequencies
        such as 200 Hz can still yield comparable performance, although they may
        require additional tuning of control parameters. For Dynamixel servos to
        support a 500 Hz control frequency, ensure that the Baud Rate is set to 4 Mbps
        and the Return Delay Time is set to 0 using the Dynamixel Wizard software.
        """
        # Gate control until manual calibration completes
        if not self.control_enabled:
            return

        if now is None:
            now = time.perf_counter()

        # Handle SYNCING state - use position control to align with robot
        if self._teleop_state == TeleopState.SYNCING:
            sync_complete, sync_pos, sync_gripper = self._run_sync_step()
            self._last_leader_pos = sync_pos
            if sync_complete:
                self._teleop_state = TeleopState.ACTIVE
                self.logger.info("[STATE] -> ACTIVE: Synced to robot.")
            self.update_communication(sync_pos, sync_gripper, now)
            return

        # Get current joint states
        leader_arm_pos, leader_arm_vel, leader_gripper_pos, leader_gripper_vel = self.get_leader_joint_states()
        self._last_leader_pos = leader_arm_pos

        # Compute control torques
        torque_arm = np.zeros(self.num_arm_joints)
        torque_l, torque_gripper = self.joint_limit_barrier(
            leader_arm_pos, leader_arm_vel, leader_gripper_pos, leader_gripper_vel
        )
        torque_arm += torque_l
        torque_arm += self.null_space_regulation(leader_arm_pos, leader_arm_vel)

        if self.enable_gravity_comp:
            torque_arm += self.gravity_compensation(leader_arm_pos, leader_arm_vel)
            torque_arm += self.friction_compensation(leader_arm_vel)

        if self.enable_torque_feedback:
            external_joint_torque = self.get_leader_arm_external_joint_torque()
            torque_arm += self.torque_feedback(external_joint_torque, leader_arm_vel)

        if self.enable_gripper_feedback:
            gripper_feedback = self.get_leader_gripper_feedback()
            torque_gripper += self.gripper_feedback(leader_gripper_pos, leader_gripper_vel, gripper_feedback)

        self.set_leader_joint_torque(torque_arm, torque_gripper)
        self.update_communication(leader_arm_pos, leader_gripper_pos, now)

    def get_leader_arm_external_joint_torque(self):
        """
        Get external joint torque from follower arm via ZMQ subscription.
        Returns zero torque if feedback is disabled or no message received.
        """
        if self.franka_torque_sub is not None and self.franka_torque_sub.message is not None:
            return np.array(self.franka_torque_sub.message)[:7]
        else:
            return np.zeros(self.num_arm_joints)

    def get_leader_gripper_feedback(self):
        """
        This method should retrieve any data from the follower gripper that might be required
        to achieve force-feedback in the leader gripper. For example, this method can be used
        to get the current position of the follower gripper for position-position force-feedback
        or the current force of the follower gripper for position-force force-feedback in the
        leader gripper. This method is called at every iteration of the control loop if
        self.enable_gripper_feedback is set to True.

        Returns:
            Any: Feedback data required by the leader gripper. This can be a NumPy array, a
            scalar, or any other data type depending on the implementation.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        if self.franka_torque_sub is not None and self.franka_torque_sub.message is not None:
            return np.array(self.franka_torque_sub.message)[-1]
        else:
            return np.zeros(1)

    def gripper_feedback(self, leader_gripper_pos, leader_gripper_vel, gripper_feedback):
        """
        Processes feedback data from the follower gripper. This method is intended to compute
        force-feedback for the leader gripper. This method is called at every iteration of the
        control loop if self.enable_gripper_feedback is set to True.

        Args:
            leader_gripper_pos (float): Leader gripper position. Can be used to provide force-
            feedback for the gripper.
            leader_gripper_vel (float): Leader gripper velocity. Can be used to provide force-
            feedback for the gripper.
            gripper_feedback (Any): Feedback data from the gripper. The format can vary depending
            on the implementation, such as a NumPy array, scalar, or custom object.

        Returns:
            float: The computed joint torque value to apply force-feedback to the leader gripper.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        deadzone = 0.01
        torque_limit = 0.075
        force_feedback = 0.0
        throttled_velocity = np.clip(leader_gripper_vel, -0.25, 0.25)
        damping = self.gripper_feedback_damping * throttled_velocity
        gripper_position_lowpass = leader_gripper_pos * 0.2 + self.gripper_pos_prev * 0.8
        # Virtual spring-damper feedback based on local gripper state
        if leader_gripper_pos > deadzone:
            force_feedback -= min(self.gripper_feedback_gain * (leader_gripper_pos - deadzone), torque_limit) + damping
        self.gripper_pos_prev = gripper_position_lowpass
        return force_feedback

    def update_communication(self, leader_arm_pos, leader_gripper_pos, now: Optional[float] = None):
        """
        Publish joint states via ZMQ at a throttled rate (50 Hz) to avoid flooding.
        Sends 9 values: 7 arm joints + 1 gripper position + 1 sync status.
        Sync status: 0=IDLE, 1=SYNCING, 2=SYNC_COMPLETE (consensus signal to policy)
        Also tracks policy connection state based on received messages.
        """
        if now is None:
            now = time.perf_counter()
        if now - self._last_cmd_publish_time >= self._cmd_publish_interval:
            try:
                # Send arm position + gripper position + sync_status (9 values total)
                self._state_msg_buf[:7] = leader_arm_pos
                self._state_msg_buf[7] = leader_gripper_pos
                self._state_msg_buf[8] = self._sync_status
                self.franka_cmd_pub.send_message(self._state_msg_buf)
            except Exception as e:
                self.logger.error(f"ZMQ send error: {e}")
            self._last_cmd_publish_time = now

        # Keep a local copy of the latest franka follower state if available
        self.latest_franka_state = getattr(self.franka_joint_state_sub, "message", None)

        # Track policy connection state
        if self.latest_franka_state is not None:
            if not self._policy_connected:
                self._policy_connected = True
                self.logger.info("[ZMQ] Policy connected - receiving robot state messages")
            self._last_policy_msg_time = now
        elif self._policy_connected and (now - self._last_policy_msg_time) > 2.0:
            # Lost connection - no messages for 2 seconds
            self._policy_connected = False
            self.logger.warning("[ZMQ] Policy disconnected - no messages received for 2s")
            # Reset to IDLE state when policy disconnects
            if self._teleop_state != TeleopState.IDLE:
                self._teleop_state = TeleopState.IDLE
                self._sync_status = 0.0
                self.logger.info("[STATE] -> IDLE: Policy disconnected")

        # Process commands from policy (9th element of message)
        self._process_policy_commands()

    # lifecycle: start / stop the high-frequency loop
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self.logger.info("Control loop started.")

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.driver.set_torque_mode(False)
        except Exception:
            pass
        self.logger.info("DIRECTTeleop control loop stopped.")

    def _run_loop(self):
        next_time = time.perf_counter()
        self.logger.info("Starting control loop...")
        while self._running:
            try:
                now = time.perf_counter()
                if now < next_time:
                    time.sleep(next_time - now)
                self.control_loop_callback(now)
            except Exception as e:
                self.logger.error(f"Control loop error: {e}")
                import traceback

                print(traceback.format_exc())
            finally:
                next_time += self.dt

    def _process_policy_commands(self):
        """
        Process commands from the policy sent as the 9th element of joint state messages.
        Implements state machine transitions based on received commands.
        DIRECT maintains its own sync_status which it publishes to policy.
        """
        msg = self.latest_franka_state
        if msg is None:
            return

        if len(msg) < 9:
            return

        # Extract command from 9th element (index 8)
        cmd_code = round(msg[8])
        robot_joint_pos = np.array(msg[:7])
        robot_gripper_pos = msg[7] if len(msg) > 7 else 0.0

        try:
            cmd = TeleopCommand(cmd_code)
        except ValueError:
            cmd = TeleopCommand.NONE

        if cmd == TeleopCommand.MIRROR:
            if self._teleop_state == TeleopState.IDLE:
                # Transition to SYNCING state
                self._teleop_state = TeleopState.SYNCING
                self._target_robot_pos = robot_joint_pos.copy()
                self._target_robot_gripper = robot_gripper_pos
                self._sync_status = 1.0  # Signal SYNCING
                self._sync_stable_frames = 0  # Reset stability counter
                self._reset_sync_patience()
                self.logger.info(f"[STATE] IDLE -> SYNCING: Target={np.round(robot_joint_pos, 3)}")
                self._start_sync_display()
            elif self._teleop_state == TeleopState.ACTIVE:
                # Check if we've drifted too far (hysteresis drift detection)
                # Use position cached this cycle in control_loop_callback to avoid an extra hardware read
                pos_error = np.linalg.norm(self._last_leader_pos - robot_joint_pos)

                if pos_error > self._sync_pos_drift_threshold:
                    # Auto-sync: transition from ACTIVE -> SYNCING to re-align leader arm
                    self._teleop_state = TeleopState.SYNCING
                    self._target_robot_pos = robot_joint_pos.copy()
                    self._target_robot_gripper = robot_gripper_pos
                    self._sync_status = 1.0  # Signal SYNCING
                    self._sync_stable_frames = 0  # Reset stability counter
                    self._reset_sync_patience()
                    self.logger.info(f"[STATE] ACTIVE -> SYNCING: Drift={pos_error:.4f}")
                    self._start_sync_display()
                else:
                    # Still in tolerance, just update target
                    self._target_robot_pos = robot_joint_pos.copy()
                    self._target_robot_gripper = robot_gripper_pos
            elif self._teleop_state == TeleopState.SYNCING:
                # Update target position while syncing
                self._target_robot_pos = robot_joint_pos.copy()
                self._target_robot_gripper = robot_gripper_pos

        elif cmd == TeleopCommand.DISABLE:
            # Explicitly disable movement - transition to IDLE
            if self._teleop_state != TeleopState.IDLE:
                self.logger.info("[STATE] -> IDLE: Disabled")
                self._stop_sync_display()
                self._teleop_state = TeleopState.IDLE
                self._sync_status = 0.0  # Signal IDLE

    # ------------------------------------------------------------------
    # Rich sync display helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_delta_bar(delta: float, max_delta: float = 0.5, half_width: int = 6) -> str:
        """Return a Rich-markup string with a compact centered directional bar for *delta*."""
        frac = max(-1.0, min(1.0, delta / max_delta))
        filled = round(abs(frac) * half_width)
        if frac >= 0:
            bar = " " * half_width + "┼" + "█" * filled + " " * (half_width - filled)
        else:
            bar = " " * (half_width - filled) + "█" * filled + "┼" + " " * half_width
        color = "green" if abs(frac) < 0.2 else "yellow" if abs(frac) < 0.7 else "bold red"
        return f"[{color}]{bar}[/{color}]"

    def _build_sync_table(
        self,
        curr_pos: np.ndarray,
        curr_gripper: float,
        target_pos: np.ndarray,
        target_gripper: float,
    ):
        """Build a Rich Table showing per-joint sync deltas."""
        from rich import box
        from rich.table import Table

        table = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold dim",
            padding=(0, 1),
        )
        table.add_column("J", style="bold", width=2)
        table.add_column("Curr", justify="right", width=7)
        table.add_column("Tgt", justify="right", width=7)
        table.add_column("Δ", justify="right", width=7)
        table.add_column("◄─────┼─────►", justify="left", width=13)
        table.add_column("", width=2)
        table.add_column("", width=7)

        for i in range(self.num_arm_joints):
            delta = target_pos[i] - curr_pos[i]
            in_sync = abs(delta) < self._sync_pos_threshold
            color = "green" if in_sync else ("yellow" if abs(delta) < 0.3 else "bold red")
            icon = "[green]✓[/]" if in_sync else f"[{color}]↻[/{color}]"
            # Per-joint state column
            if self._sync_kick_frames[i] > 0:
                state_cell = f"[bold yellow]⚡{self._sync_kick_frames[i]}f[/]"
            elif abs(delta) >= self._sync_pos_threshold and abs(delta) < self._sync_kick_region:
                pct = min(self._sync_joint_stagnation_frames[i], self._sync_error_stagnation_threshold)
                state_cell = f"[dim]⌛{pct}[/]"
            elif in_sync:
                state_cell = ""
            else:
                state_cell = "[dim]far[/]"
            table.add_row(
                f"{i + 1}",
                f"{curr_pos[i]:+.3f}",
                f"{target_pos[i]:+.3f}",
                f"[{color}]{delta:+.3f}[/{color}]",
                self._make_delta_bar(delta),
                icon,
                state_cell,
            )

        g_delta = target_gripper - curr_gripper
        g_sync = abs(g_delta) < self._sync_pos_threshold * 2.5
        g_color = "green" if g_sync else ("yellow" if abs(g_delta) < 0.3 else "bold red")
        table.add_row(
            "G",
            f"{curr_gripper:+.3f}",
            f"{target_gripper:+.3f}",
            f"[{g_color}]{g_delta:+.3f}[/{g_color}]",
            self._make_delta_bar(g_delta),
            "[green]✓[/]" if g_sync else f"[{g_color}]↻[/{g_color}]",
            "",
        )
        return table

    def _start_sync_display(self) -> None:
        """Create and start the Rich live sync display."""
        self._stop_sync_display()
        from rich.live import Live

        try:
            self._sync_live = Live(auto_refresh=False, redirect_stdout=True, redirect_stderr=True, transient=True)
            self._sync_live.start()
        except Exception:
            self._sync_live = None
        self._sync_heartbeat_frames = 0

    def _stop_sync_display(self) -> None:
        """Stop the Rich live sync display if running."""
        if self._sync_live is not None:
            try:
                self._sync_live.stop()
            except Exception:
                pass
            self._sync_live = None

    def _reset_sync_patience(self) -> None:
        """Reset all per-joint patience and kick tracking arrays."""
        self._sync_joint_best_error[:] = np.inf
        self._sync_joint_stagnation_frames[:] = 0
        self._sync_kick_frames[:] = 0

    def _update_sync_display(
        self,
        curr_pos: np.ndarray,
        curr_gripper: float,
        target_pos: np.ndarray,
        target_gripper: float,
    ) -> None:
        """Refresh the live panel with the latest per-joint deltas."""
        if self._sync_live is None:
            return
        from rich.panel import Panel

        table = self._build_sync_table(curr_pos, curr_gripper, target_pos, target_gripper)
        n_kicking = int(np.sum(self._sync_kick_frames > 0))
        kick_info = f"  [bold yellow] {n_kicking} joint(s) kicking[/]" if n_kicking > 0 else ""
        panel = Panel(
            table,
            title=f"[bold cyan] SYNCING[/]{kick_info}",
            border_style="cyan",
        )
        try:
            self._sync_live.update(panel, refresh=True)
        except Exception:
            pass

    def _run_sync_step(self):
        """
        Execute one sync step aligning all joints simultaneously with the robot.
        Tracks stagnation and applies kicks on a per-joint basis so settled joints
        are never disturbed. Returns (complete, curr_pos, curr_gripper_pos).
        """
        if self._target_robot_pos is None:
            self.logger.warning("[SYNC] No target robot position set!")
            self._sync_status = 1.0
            curr_pos, _, curr_gripper_pos, _ = self.get_leader_joint_states()
            return False, curr_pos, curr_gripper_pos

        curr_pos, curr_vel, curr_gripper_pos, curr_gripper_vel = self.get_leader_joint_states()

        delta = self._target_robot_pos - curr_pos  # signed per-joint error
        joint_errors = np.abs(delta)  # unsigned per-joint error
        gripper_pos_error = abs(curr_gripper_pos - self._target_robot_gripper)
        arm_vel_norm = np.linalg.norm(curr_vel)
        gripper_vel_norm = abs(curr_gripper_vel)

        # --- Stability check: ALL joints must be within threshold (per-joint, not global norm) ---
        if (
            np.all(joint_errors < self._sync_pos_threshold)
            and gripper_pos_error < self._sync_pos_threshold * 2.5
            and arm_vel_norm < self._sync_vel_threshold
            and gripper_vel_norm < self._sync_vel_threshold
        ):
            self._sync_stable_frames += 1
            if self._sync_stable_frames >= self._sync_stable_threshold:
                self._sync_status = 2.0  # SYNC_COMPLETE
                self.logger.info(
                    f"[SYNC] All joints synced  max_joint_err={joint_errors.max():.4f}  "
                    f"gripper_err={gripper_pos_error:.4f}"
                )
                self._stop_sync_display()
                return True, curr_pos, curr_gripper_pos
        else:
            self._sync_stable_frames = 0

        # --- Per-joint patience & kick tracking ---
        in_kick_region = joint_errors < self._sync_kick_region  # booleans per joint

        # Joints that left the kick region: reset their state
        outside = ~in_kick_region
        self._sync_joint_best_error[outside] = np.inf
        self._sync_joint_stagnation_frames[outside] = 0
        self._sync_kick_frames[outside] = 0

        if np.any(in_kick_region):
            # Joints that improved (error dropped by at least improvement_threshold)
            improved = in_kick_region & (
                joint_errors < self._sync_joint_best_error - self._sync_error_improvement_threshold
            )
            self._sync_joint_best_error[improved] = joint_errors[improved]
            self._sync_joint_stagnation_frames[improved] = 0
            self._sync_kick_frames[improved] = 0  # cancel kick if the joint is now making progress

            # Joints in region that did NOT improve: increment stagnation
            self._sync_joint_stagnation_frames[in_kick_region & ~improved] += 1

            # Trigger kick for joints that are stuck AND still above sync threshold
            above_threshold = joint_errors >= self._sync_pos_threshold
            should_kick = (
                in_kick_region
                & above_threshold
                & (self._sync_joint_stagnation_frames >= self._sync_error_stagnation_threshold)
                & (self._sync_kick_frames == 0)
            )
            if np.any(should_kick):
                self._sync_kick_frames[should_kick] = self._sync_kick_duration_frames
                self._sync_joint_stagnation_frames[should_kick] = 0
                kicked_names = [f"J{i + 1}" for i in range(self.num_arm_joints) if should_kick[i]]
                msg = f"[SYNC] Kick! joints: {', '.join(kicked_names)}"
                self.logger.info(msg)
                if self._sync_live is not None:
                    self._sync_live.console.print(f"[bold yellow] {msg}[/]")

        # --- PD control ---
        step = self._sync_interpolation_step
        arm_target = np.where(joint_errors > step, curr_pos + step * np.sign(delta), self._target_robot_pos)
        g_delta = self._target_robot_gripper - curr_gripper_pos
        gripper_target = (
            curr_gripper_pos + step * np.sign(g_delta) if abs(g_delta) > step else self._target_robot_gripper
        )

        torque = -self._sync_kp_gains * (curr_pos - arm_target) - self._sync_kd_gains * curr_vel

        # Apply kick torque only to joints with an active per-joint countdown
        kicking = self._sync_kick_frames > 0
        if np.any(kicking):
            torque[kicking] += self._sync_kick_torque * np.sign(delta[kicking])
            self._sync_kick_frames[kicking] -= 1

        if self.enable_gravity_comp:
            torque += self.gravity_compensation(curr_pos, curr_vel)

        gripper_torque = -self._sync_kp * (curr_gripper_pos - gripper_target) - self._sync_kd * curr_gripper_vel

        self.set_leader_joint_torque(torque, gripper_torque)
        self._sync_status = 1.0

        # Heartbeat display
        self._sync_heartbeat_frames += 1
        if self._sync_heartbeat_frames % self._sync_heartbeat_interval == 0:
            self._update_sync_display(
                curr_pos,
                curr_gripper_pos,
                self._target_robot_pos,
                self._target_robot_gripper,
            )

        return False, curr_pos, curr_gripper_pos

    def get_teleop_state(self):
        """Return the current teleop state for external queries."""
        return self._teleop_state

    def is_movement_enabled(self):
        """Return True if teleop is actively controlling the robot."""
        return self._teleop_state == TeleopState.ACTIVE

    def is_policy_connected(self):
        """Return True if the policy is connected and sending messages."""
        return self._policy_connected

    def get_status(self):
        """
        Get comprehensive status of the teleop controller.
        Returns a dict with state info for debugging/monitoring.
        """
        return {
            "teleop_state": self._teleop_state.name,
            "policy_connected": self._policy_connected,
            "movement_enabled": self.is_movement_enabled(),
            "control_enabled": self.control_enabled,
        }

    def shut_down(self):
        self.stop()
        try:
            self.set_leader_joint_torque(np.zeros(self.num_arm_joints), 0.0)
            self.driver.set_torque_mode(False)
        except Exception:
            pass
