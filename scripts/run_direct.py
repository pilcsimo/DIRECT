import argparse

from direct.direct_teleop.direct_teleop import DIRECTTeleop


def main():
    parser = argparse.ArgumentParser(description="Run DIRECT Teleop.")
    parser.add_argument("--config", type=str, default="direct_stable.yaml", help="Config file name.")
    parser.add_argument(
        "--calib", action="store_true", help="Force manual calibration even if a cached .npy file exists."
    )
    parser.add_argument(
        "--calib-file",
        type=str,
        default=None,
        help="Path to calibration file (.npy). Default is derived from the config name.",
    )
    args = parser.parse_args()

    teleop = DIRECTTeleop(
        config_file_name=args.config,
        force_recalibrate=args.calib,
        calibration_file=args.calib_file,
    )
    teleop.start()

    # Keep main thread alive so the teleop background thread (daemon) can run.
    # Exit cleanly on Ctrl-C.
    try:
        import time

        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        teleop.stop()
        try:
            teleop.shut_down()
        except Exception:
            pass


if __name__ == "__main__":
    main()
