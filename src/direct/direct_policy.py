"""
Droid Teleop Policy for DIRECT.

This policy acts as a lightweight ZMQ client, subscribing to the joint states
published by a separate, standalone DIRECTTeleop process. It provides the
necessary interface to be used as a teleoperation method within the DROID GUI.

Features:
- Subscribe to DIRECT joint positions via ZMQ
- Send control commands to DIRECT (gravity comp, reset, etc.)
- Keyboard shortcuts for success/failure signaling
"""

import time

import numpy as np
from droid.misc.subprocess_utils import run_threaded_command
from pynput import keyboard

from direct.python_utils.global_configs import franka_direct_zmq_addresses
from direct.python_utils.zmq_messenger import ZMQPublisher, ZMQSubscriber


class DIRECTTeleopPolicy:
    """
    A Droid-compatible teleoperation policy that subscribes to DIRECT's
    joint state publisher and can send control commands.
    """

    def __init__(
        self,
        demo: bool = False,
    ):
        """
        Initializes the DIRECT teleop policy.

        Args:
            demo (bool): Unused, kept for compatibility.
        """
        self._state = {}
        self.reset_state()

        joint_state_addr = franka_direct_zmq_addresses["joint_pos_cmd_pub"]
        robot_joints_addr = franka_direct_zmq_addresses["joint_state_sub"]
        robot_torques_addr = franka_direct_zmq_addresses["joint_torque_sub"]

        # Policy acts as CLIENT - connects to DIRECT's bound sockets
        self.joint_state_sub = ZMQSubscriber(joint_state_addr, bind_mode=False)
        self.franka_cmd_pub = ZMQPublisher(robot_joints_addr, bind_mode=False)
        self.franka_torque_pub = ZMQPublisher(robot_torques_addr, bind_mode=False)

        # Thresholds - looser for better sync behavior
        self.gripper_opened_threshold = 0.25  # Threshold to consider gripper "opened"
        self.gripper_closed_threshold = 0.5  # Threshold to consider gripper "closed"
        self.position_sync_threshold = 0.2  # Radians - position error threshold (matches teleop side)
        self.velocity_sync_threshold = 0.1  # Radians/sec - velocity threshold (matches teleop side)

        # ==================== DESYNC DETECTION STRIKE PARAMETERS ====================
        # Adjustable parameters for last-ditch desync safety
        self.desync_position_error_threshold = 0.5  # Radians - position error above this counts as strike
        self.desync_strike_limit = 20  # Number of consecutive high-error frames before emergency auto-sync

        self.auto_sync_enabled = True  # Enable/disable auto-sync functionality when connection issues detected

        # Internal State
        self.last_read_time = time.time()
        self.last_joint_pos = np.zeros(8)  # 7 arm joints + 1 gripper
        self.delta_joint_pos = np.zeros(8)  # Change in joint positions since last read
        self.joint_torques = np.zeros(8)  # 7 arm torques + 1 gripper torque (external feedback)

        # Initialize robot_joint_pos to zeros (updated from robot observations)
        self.robot_joint_pos = np.zeros(8)
        self._robot_pos_from_obs = False  # Track if we've received real robot position

        # Command channel: 9th element in joint state message
        # Command codes (sent as 9th element)
        self.CMD_NONE = 0.0
        self.CMD_MIRROR = 2.0
        self.CMD_SET_NULL_SPACE = 3.0
        self.CMD_DISABLE = 4.0  # Explicitly disable movement
        self._pending_command = self.CMD_NONE

        # Syncing state tracking
        self._syncing = False  # True when waiting for leader arm to sync
        self._sync_start_time = 0.0
        self._sync_timeout = 30.0  # Max time to wait for sync (seconds)
        self._auto_sync = False  # True when auto-sync was triggered (vs manual sync)
        self._sync_checkpoint = None  # Target position for sync (current teleop position to minimize robot movement)

        # Sync consensus: DIRECT publishes sync_status as 9th element, policy waits for signal + verifies comms
        self._direct_sync_status = 0.0  # 0=IDLE, 1=SYNCING, 2=SYNC_COMPLETE (from DIRECT)
        self._sync_health_check_frames = 0  # Consecutive frames with good connection during sync
        self._sync_health_threshold = 5  # Need 5 consecutive healthy frames to confirm ready
        self._last_heartbeat_time = time.time()  # For periodic status heartbeat (every 5s)

        print(f"[INFO] DIRECTTeleopPolicy initialized (auto_sync={self.auto_sync_enabled})")

        # ==================== DESYNC DETECTION STRIKE SYSTEM ====================
        # Last-ditch safety: if position error exceeds threshold for consecutive frames, trigger auto-sync
        self.desync_strikes = 0  # Current consecutive strike count
        print(
            f"[INFO] Desync detection: threshold={self.desync_position_error_threshold}rad"
            f" | limit={self.desync_strike_limit} strikes"
        )

        # ==================== FEEDBACK MODE SELECTION ====================
        # Two modes (cycle with 'f'):
        #   "off"            - force feedback disabled (sends zeros)
        #   "aligned_torque" - torques with deadband, cap, EMA smoothing, and per-joint gate:
        #                      only forces that would close the leader-follower gap are passed through
        self._feedback_modes = ["off", "aligned_torque"]
        self._feedback_mode = "off"  # Default: feedback off (safest)
        print(f"[INFO] Feedback mode: {self._feedback_mode.upper()}")

        # Force feedback: parameters for torque-based modes
        self._force_feedback_torques = np.zeros(7)
        self._force_threshold = 1.0  # Nm - deadband to filter noise
        self._force_max_sent = 5.0  # Nm - hard safety cap

        # Exponential moving average filter for torque-based modes
        # filtered = alpha * current + (1 - alpha) * previous
        self._torque_filtered = np.zeros(7)  # EMA state
        self._damping_alpha = 0.2  # 20% new value, 80% previous (strong smoothing)

        # Sync safety: disable force feedback for 1 second after sync completes
        self._sync_kickin_delay = 1.0  # Seconds before torque feedback activates post-sync
        self._last_sync_complete_time = None  # When sync completed (for delay timer)

        # Visualization: show received and sent torques periodically
        self._vis_enabled = True
        self._last_vis_time = time.time()
        self._vis_interval = 0.5  # Real-time monitoring

        # Input enable/disable toggle
        self._inputs_enabled = True  # Global toggle for all keyboard inputs

        # Connection tracking counters (initialized here to avoid hasattr checks in forward())
        self._prev_command_success_count = 0
        self._high_latency_count = 0
        # ==================== SAFETY DE-TETHER ====================
        # Hard-reset to idle if the robot position stops changing while movement is enabled.
        # _hard_reset() fully disables movement so the user must deliberately press 'm'
        # again to re-attempt after fixing the root cause.
        self._robot_pos_stagnation_timeout = 15.0  # Seconds without robot pos change → hard reset
        self._last_robot_pos_change_time = time.time()
        self._last_known_robot_pos = np.zeros(7)
        self._robot_pos_stagnation_threshold = 0.01  # Rad norm — below this counts as stagnant

        # Absolute-desync verification probe integrated into stagnation detection.
        # After stagnation timeout, inject tiny +/- variation on last arm joint and
        # verify DIRECT (leader) responds before declaring absolute desync.
        self._desync_probe_active = False
        self._desync_probe_start_time = 0.0
        self._desync_probe_phase = 0  # 0 = +delta, 1 = -delta
        self._desync_probe_baseline_joint = 0.0

        # Start keyboard listener for interactive success/failure signals
        self._kbd_listener = keyboard.Listener(on_press=self._on_key_press)
        self._kbd_listener.daemon = True
        self._kbd_listener.start()

        run_threaded_command(self._update_internal_state)

    def reset_state(self):
        """Reset internal state flags."""
        self._state = {
            "success": False,
            "failure": False,
            "movement_enabled": False,  # Start disabled, require explicit enable
            "controller_on": True,
            "syncing": False,  # True when leader arm is syncing to robot
        }

    def _process_force_feedback(self, raw_torques):
        """
        Process force feedback based on the selected mode and return torques ready for ZMQ
        transmission to DIRECT (which will negate them via tau_ff = -k * torque before applying
        to the leader arm).

        Modes:
        - "off": Force feedback disabled; always returns zeros.
        - "aligned_torque": External motor torques with deadband, safety cap, and EMA smoothing.
          A per-joint gate suppresses torques whose feedback direction would widen the
          leader-follower gap (condition: torque_i * delta_i < 0, where
          delta_i = robot_pos[i] - leader_pos[i]).

        Args:
            raw_torques: Raw external torques from the robot (7 elements).

        Returns:
            Processed torques ready for transmission (7 elements).
        """
        if self._feedback_mode == "off":
            return np.zeros(7)

        # Pre-processing: deadband, cap, gate, EMA
        torques = np.array(raw_torques, dtype=float).copy()
        # Deadband: zero out noise below threshold
        torques[np.abs(torques) < self._force_threshold] = 0.0
        # Hard safety cap
        torques = np.clip(torques, -self._force_max_sent, self._force_max_sent)

        if self._feedback_mode == "aligned_torque":
            # Per-joint gate: keep only torques whose feedback (negated by the leader via
            # tau_ff = -k * torque) acts to close the leader-follower position gap.
            # Condition: torque_i * delta_i < 0  →  aligned with closing the gap.
            delta = self.robot_joint_pos[:7] - self.last_joint_pos[:7]
            gate = (torques * delta) < 0  # True per joint where torque is "aligning"
            torques *= gate.astype(float)

        # EMA smoothing
        # filtered = alpha * current + (1 - alpha) * previous
        self._torque_filtered = self._damping_alpha * torques + (1.0 - self._damping_alpha) * self._torque_filtered
        return self._torque_filtered.copy()

    def _get_final_torques_for_zmq(self, processed_torques):
        """
        Apply safety gates before sending via ZMQ.
        Returns (torques, gate_status) where gate_status describes why torques might be zero.
        """
        # Gate 1: Movement disabled
        if not self._state.get("movement_enabled", False):
            return np.zeros(7), "movement_off"

        # Gate 2: Still syncing
        if self._syncing:
            return np.zeros(7), "syncing"

        # Gate 3: Post-sync kickin delay — allow the robot to settle before torque feedback resumes
        if self._last_sync_complete_time is not None:
            elapsed = time.time() - self._last_sync_complete_time
            if elapsed < self._sync_kickin_delay:
                return np.zeros(7), "kickin_delay"

        # All gates passed — send torques
        return processed_torques, "active"

    def _display_torques(self, received_torques, processed_torques, gate_status):
        """Display received and sent torques periodically with desync strike counter and feedback mode."""
        if not self._vis_enabled:
            return

        current_time = time.time()
        if current_time - self._last_vis_time < self._vis_interval:
            return

        self._last_vis_time = current_time

        # Format torque values
        received_str = ", ".join([f"{v:6.2f}" for v in received_torques[:7]])
        processed_str = ", ".join([f"{v:6.2f}" for v in processed_torques[:7]])

        # Status indicator
        friendly = {
            "active": "ACTIVE",
            "movement_off": "MOVEMENT_OFF",
            "syncing": "SYNCING",
            "kickin_delay": "KICKIN_DELAY",
        }
        status_indicator = friendly.get(gate_status, gate_status.upper())

        # Desync strike indicator
        strike_indicator = (
            f" | Desync Strikes: {self.desync_strikes}/{self.desync_strike_limit}" if self.desync_strikes > 0 else ""
        )

        # Feedback mode
        feedback_mode = self._feedback_mode.upper()

        print(
            f"[TORQUE] Status: {status_indicator} | Mode: {feedback_mode}"
            f" | Robot: [{received_str}] | ZMQ: [{processed_str}]{strike_indicator}"
        )

    def _update_internal_state(self, hz=50):
        """Continuously update internal state from ZMQ messages."""
        while True:
            # Regulate Read Frequency #
            time.sleep(1 / hz)

            # Override with pending command first (highest priority)
            if self._pending_command != self.CMD_NONE:
                current_cmd = self._pending_command
                self._pending_command = self.CMD_NONE
            elif self._syncing:
                # While syncing, continuously send MIRROR command to tell DIRECT to move to robot pos
                current_cmd = self.CMD_MIRROR
            else:
                # Not syncing - send NONE (DIRECT stays in current state: IDLE or ACTIVE)
                current_cmd = self.CMD_NONE

            # Send Joint Commands with command byte #
            # Important safety behavior:
            # - While syncing: send robot/checkpoint target for MIRROR operation
            # - While movement enabled: send robot target
            # - While movement disabled: hold leader at current DIRECT position so we never
            #   leak fresh robot targets into idle frames
            if self._syncing or current_cmd == self.CMD_MIRROR:
                if self._sync_checkpoint is not None:
                    send_robot_pos = self._sync_checkpoint.copy()
                else:
                    send_robot_pos = self.robot_joint_pos.copy()
            elif self._state.get("movement_enabled", False):
                send_robot_pos = self.robot_joint_pos.copy()
            else:
                send_robot_pos = self.last_joint_pos.copy()

            arm_joints = send_robot_pos[:7]
            gripper = send_robot_pos[7] if len(send_robot_pos) >= 8 else 0.0
            cmd_msg = np.append(arm_joints, gripper)

            # Append command as 9th element
            cmd_msg = np.append(cmd_msg, current_cmd)

            self.franka_cmd_pub.send_message(cmd_msg)

            # Periodic heartbeat status (every 5 seconds)
            current_time = time.time()
            if current_time - self._last_heartbeat_time >= 5.0:
                movement_status = "ON" if self._state.get("movement_enabled", False) else "OFF"
                sync_mode = "auto" if self._auto_sync else "manual"
                sync_status = f"SYNCING({sync_mode})" if self._syncing else "IDLE"
                controller_status = "ON" if self._state.get("controller_on", False) else "OFF"
                print(f"[HEARTBEAT] Movement={movement_status} | Sync={sync_status} | Controller={controller_status}")
                self._last_heartbeat_time = current_time

            # Read Joint States #
            msg = self.joint_state_sub.message
            if msg is not None:
                # Expecting a numpy array with 7 joint positions and 1 gripper value
                if len(msg) >= 7:
                    new_joint_pos = msg.copy()
                    # Calculate delta for arm joints before updating last_joint_pos
                    self.delta_joint_pos[:7] = np.abs(new_joint_pos[:7] - self.last_joint_pos[:7])
                    self.last_joint_pos[:7] = new_joint_pos[:7]
                if len(msg) >= 8:
                    new_gripper_pos = msg[7]
                    if new_gripper_pos is not None:
                        if new_gripper_pos > self.gripper_closed_threshold:
                            new_gripper_pos = 1.0  # Closed
                        elif new_gripper_pos < self.gripper_opened_threshold:
                            new_gripper_pos = 0.0  # Opened
                        else:
                            new_gripper_pos = round(new_gripper_pos, 1)  # Round intermediate values to reduce noise
                    # Calculate delta for gripper before updating last_joint_pos
                    self.delta_joint_pos[7] = abs(new_gripper_pos - self.last_joint_pos[7])
                    self.last_joint_pos[7] = new_gripper_pos  # gripper

                    # successful read of all joints = signalise controller is on
                    self._state["controller_on"] = True
                    self.last_read_time = time.time()

                    # SYNC CONSENSUS: DIRECT signals when its arm has converged, policy verifies connection is working
                    if self._syncing:
                        # Extract sync_status from DIRECT (9th element, index 8)
                        if len(msg) >= 9:
                            self._direct_sync_status = msg[8]

                        if self._direct_sync_status == 2.0:
                            # DIRECT signals sync complete on its side (arm converged)
                            # Policy just needs to verify connection is currently healthy (can exchange messages)
                            if self._prev_command_success_count == 0 and self._high_latency_count == 0:
                                # Connection is working right now - sync is complete
                                print("✓ [SYNC] Connection verified, completing sync")
                                self._complete_sync()
                            else:
                                # Connection not healthy, stay syncing
                                print("[SYNC] DIRECT ready but connection health issues detected, waiting...")
                        elif self._direct_sync_status < 2.0:
                            # DIRECT still syncing on its side (arm not yet converged)
                            pass

                        # Timeout: hard reset if the arm never converged — do NOT enable movement
                        if (time.time() - self._sync_start_time) > self._sync_timeout:
                            sync_type = "auto-sync" if self._auto_sync else "manual sync"
                            elapsed = time.time() - self._sync_start_time
                            self._hard_reset(reason=f"{sync_type} timed out after {elapsed:.0f}s — arm did not converge")
                else:
                    # No message received yet - that's fine, just continue
                    continue
            else:
                # No message from DIRECT - check if disconnected
                time_since_last_msg = time.time() - self.last_read_time
                if time_since_last_msg > 1.0:
                    # No messages for >1 second - DIRECT disconnected
                    # Force safe state: disable movement and exit any active sync
                    if self._state.get("movement_enabled", False) or self._syncing:
                        self._state["movement_enabled"] = False
                        self._syncing = False
                        self._state["syncing"] = False
                        self._auto_sync = False
                        self._sync_checkpoint = None
                        self._pending_command = self.CMD_NONE
                        self._reset_desync_probe()
                        print(f"[DISCONNECT] No messages from DIRECT for {time_since_last_msg:.1f}s - resetting to IDLE")
                        self.last_read_time = time.time()  # Reset timer to avoid repeated messages

    def _reset_desync_probe(self):
        """Reset integrated absolute-desync probe state."""
        self._desync_probe_active = False
        self._desync_probe_start_time = 0.0
        self._desync_probe_phase = 0
        self._desync_probe_baseline_joint = float(self.last_joint_pos[6])

    def _hard_reset(self, reason=""):
        """
        Full safety de-tether: disable movement, cancel any sync, and return to a
        completely clean idle state.  Used when the robot is unresponsive or a sync
        attempt has timed out.  The user must press 'm' to attempt a fresh sync.
        """
        was_active = self._state.get("movement_enabled", False) or self._syncing

        self._state["movement_enabled"] = False
        self._syncing = False
        self._state["syncing"] = False
        self._auto_sync = False
        self._sync_checkpoint = None
        self._pending_command = self.CMD_DISABLE
        self._prev_command_success_count = 0
        self._high_latency_count = 0
        self.desync_strikes = 0
        self._last_sync_complete_time = None
        self._torque_filtered = np.zeros(7)  # Clear EMA state
        self._reset_desync_probe()
        # Reset stagnation timer so the check doesn't immediately re-fire after recovery
        self._last_robot_pos_change_time = time.time()
        self._last_known_robot_pos = self.robot_joint_pos[:7].copy()

        if was_active:
            print(
                f"[SAFETY DE-TETHER] Hard reset | reason={reason} | movement=DISABLED"
                f" | action=press 'm' to resync (if E-stop on robot, restart NUC controller stack)"
            )

    def _start_sync(self, auto_sync=False, target_pos=None):
        """Start sync process (manual or automatic)."""
        self._reset_desync_probe()
        self._syncing = True
        self._sync_start_time = time.time()
        self._state["syncing"] = True
        self._auto_sync = auto_sync
        self._sync_health_check_frames = 0  # Reset health check counter

        # CRITICAL: Disable movement during ANY sync operation (manual or auto)
        # This prevents the policy from sending joint positions while the leader arm is syncing
        self._state["movement_enabled"] = False

        if target_pos is not None:
            self._sync_checkpoint = np.array(target_pos, dtype=float).copy()
            checkpoint_pos = np.round(self._sync_checkpoint[:7], 3)
            print(f"[SYNC] Starting target sync to checkpoint: {checkpoint_pos}")
        elif auto_sync:
            # For auto-sync, use checkpoint position (already set by caller)
            if self._sync_checkpoint is None:
                # Defensive fallback: if caller forgot checkpoint, use current robot pose.
                self._sync_checkpoint = self.robot_joint_pos.copy()
                print("[AUTO-SYNC] Warning: checkpoint missing, using current robot pose as fallback")
            checkpoint_pos = np.round(self._sync_checkpoint[:7], 3)
            print(f"[AUTO-SYNC] Starting auto-sync to checkpoint: {checkpoint_pos} (movement disabled)")
        else:
            # For manual sync, sync to ROBOT's current position (teleop will move TO robot)
            self._sync_checkpoint = self.robot_joint_pos.copy()
            checkpoint_pos = np.round(self._sync_checkpoint[:7], 3)
            print(
                f"[MANUAL SYNC] Starting manual sync - teleop will move to current robot position:"
                f" {checkpoint_pos} (movement disabled)"
            )

        # Send MIRROR command immediately
        self._pending_command = self.CMD_MIRROR

    def _complete_sync(self):
        """Complete sync process and return to normal operation."""
        was_auto_sync = self._auto_sync

        # Reset sync state
        self._syncing = False
        self._state["syncing"] = False
        self._auto_sync = False
        self._sync_checkpoint = None
        self._sync_health_check_frames = 0  # Reset health check counter

        # Reset desync strike counter on successful sync
        self.desync_strikes = 0

        self._state["movement_enabled"] = True
        self._pending_command = self.CMD_NONE

        # Start force feedback kickin delay timer
        self._last_sync_complete_time = time.time()

        sync_type = "auto-sync" if was_auto_sync else "manual sync"
        print(
            f"[SYNC] {sync_type} complete | movement=enabled"
            f" | feedback_mode={self._feedback_mode.upper()} | kickin_delay={self._sync_kickin_delay:.2f}s"
        )

    def _pulse_state(self, key, duration=0.1):
        """Briefly set a state flag to True, then reset it after duration."""

        def _pulse():
            self._state[key] = True
            print(f"[INFO] {key.capitalize()} triggered (pulse)")
            time.sleep(duration)
            self._state[key] = False

        import threading

        threading.Thread(target=_pulse, daemon=True).start()

    def _on_key_press(self, key):
        """Handle keyboard input for control commands:
        - 'f5': toggle all inputs ON/OFF (disable to prevent accidental commands)
        - 'a': pulse success (on for 0.1s, then off)
        - 'b': pulse failure (on for 0.1s, then off)
        - 'm': toggle movement_enabled (enable/disable sending joint cmds to robot)
        - 'n': set null-space target to current position
        - 'v': toggle torque visualization on/off
        - 'f': cycle feedback mode: OFF → ALIGNED_TORQUE → OFF
        """
        try:
            # Handle F5 as special key (function key) - always processed regardless of input state
            if hasattr(key, "name") and key.name == "f5":
                self._inputs_enabled = not self._inputs_enabled
                status = "ENABLED" if self._inputs_enabled else "DISABLED"
                print(f"[INPUT STATE] keyboard_inputs={status}")
                return

            # If inputs are disabled, ignore all other keys
            if not self._inputs_enabled:
                return

            # Regular character keys
            if not (hasattr(key, "char") and key.char is not None):
                return
            k = key.char.lower()
            if k == "a":
                self._pulse_state("success", duration=0.1)
            elif k == "b":
                self._pulse_state("failure", duration=0.1)
            elif k == "m":
                current_enabled = self._state.get("movement_enabled", False)
                if not current_enabled and not self._syncing:
                    # Safety policy: always synchronize before enabling movement.
                    # This avoids first-frame command leakage from stale/unsynced state.
                    leader_pos = self.last_joint_pos[:7]
                    robot_pos = self.robot_joint_pos[:7]
                    pos_error = np.linalg.norm(leader_pos - robot_pos)
                    print(f"[INFO] Enabling requested. Mandatory sync first (current pos_error={pos_error:.3f}rad)")
                    self._start_sync(auto_sync=False)
                elif current_enabled:
                    # Disabling movement - send DISABLE command to DIRECT
                    self._state["movement_enabled"] = False
                    self._syncing = False
                    self._state["syncing"] = False
                    self._auto_sync = False
                    self._sync_checkpoint = None
                    self._reset_desync_probe()
                    self._pending_command = self.CMD_DISABLE  # Tell DIRECT to go back to IDLE
                    print("[INFO] Movement disabled - sending DISABLE command to DIRECT")
                else:
                    print("[INFO] Already syncing, please wait...")
            elif k == "n":
                # Set null-space target to current robot position
                self._pending_command = self.CMD_SET_NULL_SPACE
                print(
                    f"[INFO] Sent SET NULL-SPACE TARGET command to current position: {np.round(self.robot_joint_pos, 3)}"
                )
            elif k == "v":
                # Toggle torque visualization
                self._vis_enabled = not self._vis_enabled
                status = "ON" if self._vis_enabled else "OFF"
                print(f"[INFO] Torque visualization {status}")
            elif k == "f":
                # Cycle through feedback modes: off → aligned_torque → off
                idx = (self._feedback_modes.index(self._feedback_mode) + 1) % len(self._feedback_modes)
                self._feedback_mode = self._feedback_modes[idx]
                # Reset EMA filter state on mode switch to avoid carry-over from previous mode
                self._torque_filtered = np.zeros(7)
                print(
                    f"[INFO] feedback_mode={self._feedback_mode.upper()} | "
                    f"deadband={self._force_threshold:.2f}Nm | cap={self._force_max_sent:.2f}Nm"
                    f" | ema_alpha={self._damping_alpha:.2f}"
                )
        except Exception:
            pass  # ignore special keys or unexpected errors

    def get_info(self):
        """Get controller state info (expected by droid interface)."""
        # Check if we are receiving messages
        self._state["controller_on"] = (time.time() - self.last_read_time) < 5  # 5-second timeout
        self._state["syncing"] = self._syncing
        return self._state

    def forward(self, obs_dict, include_info=False):
        """Get the latest joint positions from DIRECT."""

        # SAFETY ENFORCEMENT: Never allow movement_enabled during ANY sync operation
        # This is a redundant check to ensure the teleop never sends commands while syncing
        if self._syncing and self._state.get("movement_enabled", False):
            self._state["movement_enabled"] = False
            print("[SAFETY WARNING] Movement was enabled during sync - forcing disabled!")

        # Extract robot_state from nested obs_dict structure
        # obs_dict["robot_state"] contains the actual robot state dictionary
        robot_state = obs_dict.get("robot_state", obs_dict)  # Fallback to obs_dict if not nested

        # CHECK FOR HIGH TORQUE DESYNC CONDITION
        # If computed torques exceed 20 Nm, immediately trigger auto-sync for emergency recovery
        joint_torques_computed = robot_state.get("joint_torques_computed", np.zeros(7))
        if joint_torques_computed is not None:
            try:
                max_computed_torque = np.max(np.abs(np.array(joint_torques_computed)[:7]))
                if max_computed_torque > 20.0 and not self._syncing and self._state.get("movement_enabled", False):
                    self._sync_checkpoint = self.robot_joint_pos.copy()
                    print(f"[HIGH TORQUE ALERT] tau={max_computed_torque:.2f}Nm > 20.00Nm | action=emergency auto-sync")
                    self._start_sync(auto_sync=True)
            except Exception:
                pass  # Ignore if we can't process torques

        # Extract external torques for force feedback.
        # NOTE: "motor_torques_external" is not present in the standard DROID robot_state.
        # Without it, torque feedback is silently disabled (zeros). To enable it,
        # publish motor_torques_external in your DROID robot_state.
        raw_arm_torques = robot_state.get("motor_torques_external", np.zeros(7))
        raw_ext_torques = (
            np.array(raw_arm_torques, dtype=float)[:7]
            if raw_arm_torques is not None and len(raw_arm_torques) >= 7
            else np.zeros(7)
        )
        self.joint_torques = np.zeros(8)
        self.joint_torques[:7] = raw_ext_torques

        # DESYNC DETECTION: track command success and latency to detect disconnection
        cmd_successful = robot_state.get("prev_command_successful", True)
        latency_ms = robot_state.get("prev_controller_latency_ms", 0)

        # Track command failures
        if not cmd_successful:
            self._prev_command_success_count += 1
        else:
            self._prev_command_success_count = 0

        # Track high latency spikes (threshold: >2ms indicates loss of responsiveness)
        if latency_ms > 2:
            self._high_latency_count += 1
        else:
            self._high_latency_count = 0

        # Detect connection loss during active operation: trigger auto-sync if experiencing issues
        if (
            (self._prev_command_success_count > 10 or self._high_latency_count > 10)
            and self._state.get("movement_enabled", False)
            and not self._syncing
        ):
            # Persistent connection problems detected while moving - auto-sync for recovery
            self._sync_checkpoint = self.robot_joint_pos.copy()
            self._sync_checkpoint[7] = 0.0  # Force gripper open
            checkpoint_pos = np.round(self._sync_checkpoint[:7], 3)
            print(f"[DESYNC-AUTO-SYNC] Connection problems detected, teleop->robot: {checkpoint_pos}")
            self._start_sync(auto_sync=True)

        # Get robot joint positions from robot_state (not top-level obs_dict)
        raw_joint_pos = robot_state.get("joint_positions", None)
        gripper_pos = robot_state.get("gripper_position", 0.0)

        if raw_joint_pos is not None:
            # Convert to numpy array (handles lists, tuples, etc.)
            arm_joints = np.array(raw_joint_pos, dtype=np.float64)[:7]  # Ensure 7 elements
            # Update robot_joint_pos with real values
            self.robot_joint_pos[:7] = arm_joints
            self.robot_joint_pos[7] = float(gripper_pos) if gripper_pos is not None else 0.0
            self._robot_pos_from_obs = True

            # Safety de-tether trigger 2: robot position stagnant while movement is enabled.
            # If the robot stops responding to commands (e.g. controller crashed) but comms
            # are still alive, the position will freeze while the leader keeps moving freely.
            if np.linalg.norm(arm_joints - self._last_known_robot_pos) > self._robot_pos_stagnation_threshold:
                # Robot moved — reset stagnation timer
                self._last_robot_pos_change_time = time.time()
                self._last_known_robot_pos = arm_joints.copy()
                self._reset_desync_probe()
            elif self._state.get("movement_enabled", False) and not self._syncing:
                stagnation_time = time.time() - self._last_robot_pos_change_time
                if stagnation_time > self._robot_pos_stagnation_timeout:
                    # Integrated absolute-desync probe:
                    # if robot seems stagnant, inject tiny variation on DIRECT last joint and
                    # verify DIRECT responds before declaring absolute desync.
                    if not self._desync_probe_active:
                        self._desync_probe_active = True
                        self._desync_probe_start_time = time.time()
                        self._desync_probe_phase = 0
                        self._desync_probe_baseline_joint = float(self.last_joint_pos[6])
                        print(
                            f"[DESYNC-PROBE] Starting integrated probe on DIRECT joint {6 + 1} (delta=+/-{0.03:.3f} rad)"
                        )

                    probe_delta = abs(
                        float(self.last_joint_pos[6]) - float(self._desync_probe_baseline_joint)
                    )
                    if probe_delta > 0.006:
                        print(
                            f"[DESYNC-PROBE] PASS: DIRECT responded (delta={probe_delta:.4f} rad > {0.006:.4f} rad)"
                        )
                        self._last_robot_pos_change_time = time.time()
                        self._last_known_robot_pos = arm_joints.copy()
                        self._reset_desync_probe()
                    else:
                        probe_elapsed = time.time() - self._desync_probe_start_time
                        if self._desync_probe_phase == 0 and probe_elapsed >= 0.35:
                            self._desync_probe_phase = 1
                            print("[DESYNC-PROBE] No response on +delta; trying -delta")

                        if probe_elapsed >= (2.0 * 0.35):
                            self._hard_reset(
                                reason=(
                                    f"robot position stagnant for {stagnation_time:.0f}s and "
                                    f"DIRECT did not respond to integrated probe"
                                )
                            )
        else:
            # No robot state in this observation - keep using current robot_joint_pos (home or last known)
            pass

        # ==================== DESYNC DETECTION STRIKE SYSTEM ====================
        # Last-ditch safety: count consecutive frames where position error > threshold
        # After N strikes, automatically trigger sync
        if self._state.get("movement_enabled", False) and not self._syncing:
            leader_pos = self.last_joint_pos[:7]
            robot_pos = self.robot_joint_pos[:7]
            pos_error = np.linalg.norm(leader_pos - robot_pos)

            if pos_error > self.desync_position_error_threshold:
                self.desync_strikes += 1
                # Periodically display strike progress
                if self.desync_strikes % 20 == 0:
                    print(
                        f"[DESYNC-STRIKE] {self.desync_strikes}/{self.desync_strike_limit}"
                        f" | pos_error={pos_error:.3f}rad"
                    )

                # Trigger auto-sync if strike limit reached
                if self.desync_strikes >= self.desync_strike_limit:
                    self._sync_checkpoint = self.robot_joint_pos.copy()
                    self._sync_checkpoint[7] = 0.0  # Force gripper open
                    print("[DESYNC-STRIKE] Limit reached! Triggering emergency auto-sync")
                    self._start_sync(auto_sync=True)
                    self.desync_strikes = 0  # Reset after triggering
            else:
                # Position error within threshold - decay strike counter
                self.desync_strikes = max(0, self.desync_strikes - 1)

        # Single feedback processing call
        self._last_received_torques = raw_ext_torques.copy()
        self._force_feedback_torques = self._process_force_feedback(raw_ext_torques)

        # Get final torques with gate status for display
        temp_processed, gate_status = self._get_final_torques_for_zmq(self._force_feedback_torques)

        # Display received and sent torques periodically
        self._display_torques(self._last_received_torques, temp_processed, gate_status)

        # Send filtered torque command at 50Hz
        send_torque = np.append(temp_processed.copy(), 0.0)  # 8 elements
        self.franka_torque_pub.send_message(send_torque)

        action = self.last_joint_pos.copy()

        # During integrated desync probe, inject tiny variation on last DIRECT arm joint.
        if self._desync_probe_active and self._state.get("movement_enabled", False) and not self._syncing:
            probe_action = action.copy()
            j = 6
            sign = 1.0 if self._desync_probe_phase == 0 else -1.0
            probe_action[j] = self._desync_probe_baseline_joint + (sign * 0.03)
            action = probe_action

        if include_info:
            info_dict = {"target_joint_position": action[:7], "target_gripper_position": action[-1]}
            return action, info_dict

        return action
