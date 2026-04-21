
# Setting up Dynamixel Servos

> **Attribution:** The servo setup instructions below are adapted directly from the
> [FACTR Teleop README](https://github.com/JasonJZLiu/FACTR_Teleop/blob/main/src/factr_teleop/README.md)
> (© FACTR authors, Apache 2.0). Follow that guide through the
> **"Setting up Dynamixel Servos"** section (servo IDs, baud rate, and USB latency).
> **Stop before "Preparing to Launch FACTR Teleop"** — DIRECT uses a different
> calibration and launch procedure. Return to the [main README](../../README.md)
> for those steps.

Before using DIRECT, the leader arm's Dynamixel servos need to be properly
configured. Please install the [Dynamixel Wizard](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/)
software.

## Update Servo IDs

By default, all Dynamixel servos are set to ID 1. To control multiple servos with a single U2D2 controller,
each servo must be assigned a unique ID. This assignment should be done one servo at a time, starting with
the servo closest to the robot base and progressing up the kinematic chain. Assign IDs sequentially from
1 up to 8, with the gripper trigger servo holding ID 8.

### Steps to Change a Servo's ID

1. **Connect a single servo to the U2D2 controller.**
2. **Launch the Dynamixel Wizard software.**
3. **Configure the scanning settings:**
   - In the top menu bar, go to **Tools → Options**.
   - Under **"Select port to scan"**, ensure all ports are selected.
   - Under **"Select baudrate to scan"**, choose the correct baudrate. The default is **57600 bps**. If unsure, select all available options.
   - Under **"Set ID range to scan"**, you can speed up scanning by limiting the range to **0–15**, unless the servo might already have an ID outside this range.
4. **Click the Scan icon** (magnifying glass with a gear) in the top-left corner.
   - If the servo is found, it will appear in the left panel of the window.
5. **Select the detected servo** from the panel, then click **ID** found in the middle panel of the window.
   - Select the new ID on the right panel of the window and click **Save**.
6. **Repeat** the above steps for each servo.

## Increase Servo Control Frequency

There are additional Dynamixel servo settings that must be changed in order to run the control loop at
a frequency as fast as 500Hz. The steps for each servo are as follows:

### Steps to Increase Servo Control Frequency

1. **Launch the Dynamixel Wizard software** and **scan for servos**.
2. **Select each detected servo** from the left-hand panel.
3. In the middle panel, click **"Baud Rate (Bus)"**:
   - On the right-hand panel, **set the value to 4 Mbps** and click **Save**.
4. In the middle panel, click **"Return Delay Time"**:
   - On the right-hand panel, **set the value to 0** and click **Save**.

In addition to the Dynamixel servo settings, it is crucial to minimize the latency of the USB port connected to the U2D2 controller
to achieve high-frequency control. The latency timer should be set to `1` for optimal performance.

### Steps to Check and Set USB Port Latency

1. **List connected U2D2 devices**:

   ```bash
   ls /dev/serial/by-id/
   ```

   This will show device names for each connected U2D2 controller. For example:

   ```
   usb-FTDI_USB__-__Serial_Converter_FT7WBF8S-if00-port0
   usb-FTDI_USB__-__Serial_Converter_FT951EJA-if00-port0
   ```

   Identify the one corresponding to your **leader arm**. Replace `<device-name>` with the correct name in the next steps.

2. **Resolve the full path to the USB device**:

   ```bash
   readlink -f /dev/serial/by-id/<device-name>
   ```

   This will return something like:

   ```
   /dev/ttyUSB0
   ```

3. **Check the current latency timer value**:

   ```bash
   cat /sys/bus/usb-serial/devices/<ttyUSBx>/latency_timer
   ```

   Replace `<ttyUSBx>` with the ttyUSB device name from the previous step (e.g., `ttyUSB0`).

4. **If the value is not `1`, set it manually**:

   ```bash
   echo 1 | sudo tee /sys/bus/usb-serial/devices/<ttyUSBx>/latency_timer
   ```

   > **Note:** This command requires `sudo` permissions.

   > **Note:** Everytime the U2D2 controller USB is re-plugged into the computer, this command needs to be run again to ensure the
   latency timer is 1.

### Automated Latency Check

Note that the provided leader arm node **automatically checks** the latency timer on startup.
It reads the `dynamixel_port` from the YAML config file and verifies that the latency timer is set to `1`.

If the latency is not correctly set, the script will:

- Raise an exception
- Output the exact `echo` command required to fix it

Since changing the latency timer requires `sudo` permissions, **you must run the command manually** when prompted.

---

For configuration, calibration, and usage instructions see the [main README](../../README.md).
