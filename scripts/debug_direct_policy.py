import time

from direct.direct_policy import DIRECTTeleopPolicy

# Initialize policy
policy = DIRECTTeleopPolicy()

# from DROID - franka_panda.yaml rest pose
robot_start_position = [
    -0.13935425877571106,
    -0.020481698215007782,
    -0.05201413854956627,
    -2.0691256523132324,
    0.05058913677930832,
    2.0028650760650635,
    -0.9167874455451965,
]


# Create observation dictionary matching expected format
def create_observation(joint_pos, gripper_pos=0.0):
    return {
        "robot_state": {
            "joint_positions": joint_pos,
            "gripper_position": gripper_pos,
            "prev_joint_torques_computed_safened": [0.0] * 7,
            "joint_velocities": [0.0] * 7,
        }
    }


print(f"Testing with robot at start position: {robot_start_position}")
print("Press Ctrl+C to stop")

iteration = 0
try:
    while True:
        # Create observation
        observation = create_observation(robot_start_position)

        # Get policy output
        action, info = policy.forward(observation, include_info=True)
        policy_info = policy.get_info()

        # Print every 10 iterations to avoid spam
        # if iteration % 10 == 0:
        #     print(f"\nIter {iteration}:")
        #     print(f"  Leader joint pos: {np.round(action[:7], 3)}")
        #     print(f"  Policy state: movement={policy_info.get('movement_enabled')}, "
        #           f"syncing={policy_info.get('syncing')}")
        #     print(f"  Auto-sync: strikes={policy.desync_strikes}/{policy.desync_strike_limit}")

        iteration += 1
        time.sleep(0.1)  # 10 Hz

except KeyboardInterrupt:
    print(f"\nStopped after {iteration} iterations")
