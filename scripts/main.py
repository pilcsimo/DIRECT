import argparse

from droid.robot_env import RobotEnv
from droid.user_interface.data_collector import DataCollecter
from droid.user_interface.gui import RobotGUI

parser = argparse.ArgumentParser(description="Launch DROID data collection with a chosen teleoperation controller.")

# Controller selection arguments
parser.add_argument("--left_controller", action="store_true", help="Use left oculus controller")
parser.add_argument("--direct", action="store_true", help="Use DIRECT Teleoperation device")

args = parser.parse_args()

# Make the robot env

if args.direct:
    env = RobotEnv(action_space="joint_position")
    from direct.direct_policy import DIRECTTeleopPolicy as TeleopPolicy

    policy = TeleopPolicy()
    # Make the data collector
    data_collector = DataCollecter(env=env, controller=policy)
    # Make the GUI
    user_interface = RobotGUI(robot=data_collector, right_controller=True)
else:
    env = RobotEnv()
    from droid.controllers.oculus_controller import VRPolicy

    if args.left_controller:
        controller = VRPolicy(right_controller=False)
        # Make the data collector
        data_collector = DataCollecter(env=env, controller=controller)
        # Make the GUI
        user_interface = RobotGUI(robot=data_collector, right_controller=False)
    else:
        controller = VRPolicy(right_controller=True)
        # Make the data collector
        data_collector = DataCollecter(env=env, controller=controller)
        # Make the GUI
        user_interface = RobotGUI(robot=data_collector, right_controller=True)
