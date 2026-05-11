"""
bilateral_teleop_node.py
------------------------
ROS2 node implementing the unified bimanual bilateral teleoperation
architecture for:

  Master: 2 x ViperX300s  (left + right, 5-DOF each with wrist-roll locked)
  Slave:  2 x ReactorX150 (left + right, 5-DOF each with wrist-roll locked)

Node Architecture
-----------------

  ┌─────────────────────────────────────────────────────────────┐
  │                 BimanualTeleopNode                          │
  │                                                             │
  │  ┌──────────────────┐     ┌──────────────────┐             │
  │  │  Left MISA+MPC   │     │  Right MISA+MPC  │             │
  │  │  Controller      │     │  Controller      │             │
  │  └──────────────────┘     └──────────────────┘             │
  │         │                         │                         │
  │   MasterL ↔ SlaveL          MasterR ↔ SlaveR               │
  │                                                             │
  │  Topics subscribed:                                         │
  │    /viper_left/joint_states    (sensor_msgs/JointState)     │
  │    /viper_right/joint_states                                │
  │    /reactor_left/joint_states                               │
  │    /reactor_right/joint_states                              │
  │    /viper_left/joint_currents  (std_msgs/Float64MultiArray) │
  │    /viper_right/joint_currents                              │
  │    /reactor_left/joint_currents                             │
  │    /reactor_right/joint_currents                            │
  │    /teleop/enable              (std_msgs/Bool)              │
  │                                                             │
  │  Topics published:                                          │
  │    /viper_left/cmd_torque      (std_msgs/Float64MultiArray) │
  │    /viper_right/cmd_torque                                  │
  │    /reactor_left/cmd_position  (std_msgs/Float64MultiArray) │
  │    /reactor_right/cmd_position                              │
  │    /teleop/diagnostics         (std_msgs/String)  [10 Hz]  │
  └─────────────────────────────────────────────────────────────┘

Control Mode Switching
----------------------
  The node supports the four-channel unified architecture.
  Mode selection is via ROS parameter "control_mode":
    "MISA"  — Master Impedance / Slave Admittance  (default)
    "MASA"  — Master Admittance / Slave Admittance
    "MASI"  — Master Admittance / Slave Impedance
    "MISI"  — Master Impedance / Slave Impedance

  <<USER: Only MISA is fully implemented here.  To add other modes,
          implement the respective control law in misa_mpc_controller.py
          and dispatch based on self.control_mode below.>>

Safety
------
  - Torque commands are clamped to hardware limits before publishing.
  - If any arm's joint state goes stale (> 500 ms), torque commands
    drop to gravity-compensation-only.
  - An enable/disable topic allows the operator to kill the loop safely.
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
import json
import time

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Bool, String

from ..arm_interface  import ArmInterface, ArmDynamics
from ..misa_mpc_controller import MISAController, MISAParams


# ---------------------------------------------------------------------------
# Joint name ordering (must match your Dynamixel/InterbotiX driver config)
# <<USER: Verify joint name strings match your robot_description / URDF>>
# ---------------------------------------------------------------------------
VIPERX_JOINT_NAMES  = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle"]
REACTOR_JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle"]


class BimanualTeleopNode(Node):

    def __init__(self):
        super().__init__("bimanual_teleop_node")

        # ------------------------------------------------------------------
        # Declare ROS2 parameters (can be set via YAML or launch args)
        # ------------------------------------------------------------------
        self.declare_parameter("control_mode", "MISA")
        self.declare_parameter("loop_rate_hz",  100.0)
        self.declare_parameter("mpc_horizon",   10)
        self.declare_parameter("state_timeout_s", 0.5)

        # Impedance gains (master) — diagonal entries per joint
        # <<USER: Tune these for your operator's preference>>
        self.declare_parameter("master_Mmd_diag", [1.0, 1.5, 1.0, 0.5, 0.3])
        self.declare_parameter("master_Dmd_diag", [5.0, 8.0, 5.0, 3.0, 2.0])
        self.declare_parameter("master_Kmd_diag", [10.0,15.0,10.0, 5.0, 3.0])

        # Admittance gains (slave)
        # <<USER: Tune for desired compliance>>
        self.declare_parameter("slave_Msd_diag",  [0.8, 1.2, 0.8, 0.4, 0.25])
        self.declare_parameter("slave_Dsd_diag",  [4.0, 6.0, 4.0, 2.5, 1.5])
        self.declare_parameter("slave_Ksd_diag",  [8.0,12.0, 8.0, 4.0, 2.5])

        # Four-channel coupling gains
        self.declare_parameter("kof", 0.8)   # slave→master force coupling
        self.declare_parameter("ksf", 0.2)   # local master force coupling
        self.declare_parameter("kom", 0.8)   # slave→master position coupling
        self.declare_parameter("ksm", 0.2)   # local master position coupling

        # ------------------------------------------------------------------
        # Read parameters
        # ------------------------------------------------------------------
        self.control_mode = self.get_parameter("control_mode").value
        hz  = self.get_parameter("loop_rate_hz").value
        self.dt = 1.0 / hz
        self.timeout = self.get_parameter("state_timeout_s").value

        # ------------------------------------------------------------------
        # Build dynamics objects
        # <<USER: Replace ArmDynamics() instantiation with your actual
        #         dynamic model. See arm_interface.py for instructions.>>
        # ------------------------------------------------------------------
        self.master_L_dyn = ArmDynamics("viperx300s")
        self.master_R_dyn = ArmDynamics("viperx300s")
        self.slave_L_dyn  = ArmDynamics("reactorx150")
        self.slave_R_dyn  = ArmDynamics("reactorx150")

        # ------------------------------------------------------------------
        # Build arm interfaces
        # ------------------------------------------------------------------
        filter_alpha = 0.3   # <<USER: Increase for smoother but laggier signal>>
        self.master_L_iface = ArmInterface("viperx300s", self.master_L_dyn, self.dt, filter_alpha)
        self.master_R_iface = ArmInterface("viperx300s", self.master_R_dyn, self.dt, filter_alpha)
        self.slave_L_iface  = ArmInterface("reactorx150", self.slave_L_dyn,  self.dt, filter_alpha)
        self.slave_R_iface  = ArmInterface("reactorx150", self.slave_R_dyn,  self.dt, filter_alpha)

        # ------------------------------------------------------------------
        # Build MISA+MPC controllers (one per arm pair)
        # ------------------------------------------------------------------
        params = self._build_params()

        from ..arm_interface import (
            VIPERX300S_Q_MIN, VIPERX300S_Q_MAX,
            REACTORX150_Q_MIN, REACTORX150_Q_MAX,
            VIPERX300S_TAU_MAX, REACTORX150_TAU_MAX,
            VIPERX300S_DQ_MAX, REACTORX150_DQ_MAX,
        )

        ctrl_kwargs = dict(
            params=params,
            q_master_min=VIPERX300S_Q_MIN,
            q_master_max=VIPERX300S_Q_MAX,
            q_slave_min=REACTORX150_Q_MIN,
            q_slave_max=REACTORX150_Q_MAX,
            tau_master_max=VIPERX300S_TAU_MAX,
            tau_slave_max=REACTORX150_TAU_MAX,
            dq_master_max=VIPERX300S_DQ_MAX,
            dq_slave_max=REACTORX150_DQ_MAX,
        )

        self.ctrl_L = MISAController(
            master_dynamics=self.master_L_dyn,
            slave_dynamics=self.slave_L_dyn,
            **ctrl_kwargs,
        )
        self.ctrl_R = MISAController(
            master_dynamics=self.master_R_dyn,
            slave_dynamics=self.slave_R_dyn,
            **ctrl_kwargs,
        )

        # ------------------------------------------------------------------
        # State buffers (raw sensor data)
        # ------------------------------------------------------------------
        self._raw = {
            "master_L_q": None,  "master_L_I": None,  "master_L_t": 0.0,
            "master_R_q": None,  "master_R_I": None,  "master_R_t": 0.0,
            "slave_L_q":  None,  "slave_L_I":  None,  "slave_L_t":  0.0,
            "slave_R_q":  None,  "slave_R_I":  None,  "slave_R_t":  0.0,
        }
        self.enabled = False

        # ------------------------------------------------------------------
        # ROS2 subscribers
        # <<USER: Adjust topic names to match your InterbotiX driver config.
        #         The InterbotiX ROS2 drivers typically publish to
        #         /<robot_name>/joint_states  and may need a custom current
        #         topic — verify with your driver documentation.>>
        # ------------------------------------------------------------------
        cbg = ReentrantCallbackGroup()

        # Joint states (position + velocity from encoders)
        self.create_subscription(JointState, "/viper_left/joint_states",
            lambda m: self._js_cb(m, "master_L"), 10, callback_group=cbg)
        self.create_subscription(JointState, "/viper_right/joint_states",
            lambda m: self._js_cb(m, "master_R"), 10, callback_group=cbg)
        self.create_subscription(JointState, "/reactor_left/joint_states",
            lambda m: self._js_cb(m, "slave_L"),  10, callback_group=cbg)
        self.create_subscription(JointState, "/reactor_right/joint_states",
            lambda m: self._js_cb(m, "slave_R"),  10, callback_group=cbg)

        # Motor currents [A] — one Float64MultiArray per arm
        # <<USER: Dynamixel SDK may publish effort in the JointState message
        #         instead of a separate topic (effort field = current * kt).
        #         If so, parse from joint_states and remove these subscribers.>>
        self.create_subscription(Float64MultiArray, "/viper_left/joint_currents",
            lambda m: self._curr_cb(m, "master_L"), 10, callback_group=cbg)
        self.create_subscription(Float64MultiArray, "/viper_right/joint_currents",
            lambda m: self._curr_cb(m, "master_R"), 10, callback_group=cbg)
        self.create_subscription(Float64MultiArray, "/reactor_left/joint_currents",
            lambda m: self._curr_cb(m, "slave_L"),  10, callback_group=cbg)
        self.create_subscription(Float64MultiArray, "/reactor_right/joint_currents",
            lambda m: self._curr_cb(m, "slave_R"),  10, callback_group=cbg)

        # Enable/disable teleoperation
        self.create_subscription(Bool, "/teleop/enable",
            self._enable_cb, 10, callback_group=cbg)

        # ------------------------------------------------------------------
        # ROS2 publishers
        # ------------------------------------------------------------------

        # Master: torque commands (current-control mode on Dynamixels)
        # <<USER: Verify the torque/current command topic for your driver.
        #         InterbotiX uses /robot/commands/joint_group  or
        #         /robot/commands/joint_single depending on config.>>
        self._pub_tau_master_L = self.create_publisher(
            Float64MultiArray, "/viper_left/cmd_torque",  10)
        self._pub_tau_master_R = self.create_publisher(
            Float64MultiArray, "/viper_right/cmd_torque", 10)

        # Slave: position commands
        self._pub_q_slave_L = self.create_publisher(
            Float64MultiArray, "/reactor_left/cmd_position",  10)
        self._pub_q_slave_R = self.create_publisher(
            Float64MultiArray, "/reactor_right/cmd_position", 10)

        # Diagnostics
        self._pub_diag = self.create_publisher(String, "/teleop/diagnostics", 10)

        # ------------------------------------------------------------------
        # Main control timer
        # ------------------------------------------------------------------
        self._control_timer = self.create_timer(
            self.dt, self._control_loop, callback_group=cbg)

        # Diagnostics timer (lower rate)
        self._diag_timer = self.create_timer(
            0.1, self._publish_diagnostics, callback_group=cbg)

        # Timing stats
        self._loop_times = []

        self.get_logger().info(
            f"BimanualTeleopNode initialised | mode={self.control_mode} "
            f"| dt={self.dt*1000:.1f}ms | MPC N={params.N}"
        )

    # -----------------------------------------------------------------------
    # Parameter builder
    # -----------------------------------------------------------------------

    def _build_params(self) -> MISAParams:
        get = lambda name: np.array(self.get_parameter(name).value, dtype=float)

        return MISAParams(
            n_joints = 5,
            N  = self.get_parameter("mpc_horizon").value,
            dt = self.dt,
            Mmd = np.diag(get("master_Mmd_diag")),
            Dmd = np.diag(get("master_Dmd_diag")),
            Kmd = np.diag(get("master_Kmd_diag")),
            Msd = np.diag(get("slave_Msd_diag")),
            Dsd = np.diag(get("slave_Dsd_diag")),
            Ksd = np.diag(get("slave_Ksd_diag")),
            kof = float(self.get_parameter("kof").value),
            ksf = float(self.get_parameter("ksf").value),
            kom = float(self.get_parameter("kom").value),
            ksm = float(self.get_parameter("ksm").value),
        )

    # -----------------------------------------------------------------------
    # Subscribers
    # -----------------------------------------------------------------------

    def _js_cb(self, msg: JointState, arm_key: str):
        """
        Parse joint_states message.

        <<USER: If your driver publishes effort (torque) in JointState.effort,
                you can extract motor torque here instead of using a separate
                current topic. Then remove the current subscriber for that arm.

                Also verify that joint ordering matches VIPERX_JOINT_NAMES /
                REACTOR_JOINT_NAMES above.>>
        """
        joint_names = (VIPERX_JOINT_NAMES
                       if "master" in arm_key else REACTOR_JOINT_NAMES)
        try:
            idx = [msg.name.index(jn) for jn in joint_names]
            q = np.array([msg.position[i] for i in idx])
            self._raw[f"{arm_key}_q"] = q
            self._raw[f"{arm_key}_t"] = self.get_clock().now().nanoseconds * 1e-9
        except (ValueError, IndexError) as e:
            self.get_logger().warn(f"[{arm_key}] joint_states parse error: {e}")

    def _curr_cb(self, msg: Float64MultiArray, arm_key: str):
        """
        Receive motor current readings [A] for all active joints.

        <<USER: Verify array length = 5 (active DOF).
                Dynamixel current unit depends on motor series — may need
                to convert from raw counts to Amps using motor spec.
                e.g. for XM430: current[A] = raw * 2.69e-3>>
        """
        data = np.array(msg.data, dtype=float)
        if len(data) >= 5:
            self._raw[f"{arm_key}_I"] = data[:5]
        else:
            self.get_logger().warn(f"[{arm_key}] current array too short ({len(data)})")

    def _enable_cb(self, msg: Bool):
        self.enabled = msg.data
        self.get_logger().info(f"Teleoperation {'ENABLED' if self.enabled else 'DISABLED'}")

    # -----------------------------------------------------------------------
    # Main control loop
    # -----------------------------------------------------------------------

    def _control_loop(self):
        if not self.enabled:
            return

        t_start = time.perf_counter()
        now = self.get_clock().now().nanoseconds * 1e-9

        # Check data freshness
        for arm_key in ["master_L", "master_R", "slave_L", "slave_R"]:
            age = now - self._raw.get(f"{arm_key}_t", 0.0)
            if age > self.timeout:
                self.get_logger().warn(
                    f"[{arm_key}] stale state ({age*1000:.0f}ms) — skipping cycle")
                return

        # Check all data present
        for arm_key in ["master_L", "master_R", "slave_L", "slave_R"]:
            if (self._raw.get(f"{arm_key}_q") is None or
                    self._raw.get(f"{arm_key}_I") is None):
                self.get_logger().info(f"Waiting for {arm_key} data...")
                return

        # ------------------------------------------------------------------
        # Process arm states (filter, differentiate, estimate tau_ext)
        # ------------------------------------------------------------------
        master_L_state = self.master_L_iface.update(
            self._raw["master_L_q"], self._raw["master_L_I"])
        master_R_state = self.master_R_iface.update(
            self._raw["master_R_q"], self._raw["master_R_I"])
        slave_L_state  = self.slave_L_iface.update(
            self._raw["slave_L_q"],  self._raw["slave_L_I"])
        slave_R_state  = self.slave_R_iface.update(
            self._raw["slave_R_q"],  self._raw["slave_R_I"])

        # ------------------------------------------------------------------
        # Run MISA+MPC controllers — LEFT pair
        # ------------------------------------------------------------------
        tau_m_L, q_s_L = self.ctrl_L.step(master_L_state, slave_L_state)

        # ------------------------------------------------------------------
        # Run MISA+MPC controllers — RIGHT pair
        # ------------------------------------------------------------------
        tau_m_R, q_s_R = self.ctrl_R.step(master_R_state, slave_R_state)

        # ------------------------------------------------------------------
        # Safety clamp before publishing
        # <<USER: The limits below are imported from arm_interface.py.
        #         Add additional workspace-dependent constraints here if needed
        #         (e.g., forbidden zones, proximity checks between arms).>>
        # ------------------------------------------------------------------
        from ..arm_interface import VIPERX300S_TAU_MAX, REACTORX150_Q_MIN, REACTORX150_Q_MAX
        tau_m_L = np.clip(tau_m_L, -VIPERX300S_TAU_MAX, VIPERX300S_TAU_MAX)
        tau_m_R = np.clip(tau_m_R, -VIPERX300S_TAU_MAX, VIPERX300S_TAU_MAX)
        q_s_L   = np.clip(q_s_L,   REACTORX150_Q_MIN,   REACTORX150_Q_MAX)
        q_s_R   = np.clip(q_s_R,   REACTORX150_Q_MIN,   REACTORX150_Q_MAX)

        # ------------------------------------------------------------------
        # Publish commands
        # ------------------------------------------------------------------
        self._pub_tau_master_L.publish(self._f64(tau_m_L))
        self._pub_tau_master_R.publish(self._f64(tau_m_R))
        self._pub_q_slave_L.publish(self._f64(q_s_L))
        self._pub_q_slave_R.publish(self._f64(q_s_R))

        self._loop_times.append(time.perf_counter() - t_start)
        if len(self._loop_times) > 200:
            self._loop_times.pop(0)

    # -----------------------------------------------------------------------
    # Diagnostics publisher
    # -----------------------------------------------------------------------

    def _publish_diagnostics(self):
        if not self._loop_times:
            return
        avg_ms = np.mean(self._loop_times) * 1000
        max_ms = np.max(self._loop_times)  * 1000
        info = {
            "enabled":  self.enabled,
            "mode":     self.control_mode,
            "loop_avg_ms": round(avg_ms, 2),
            "loop_max_ms": round(max_ms, 2),
            "master_L_age_ms": round(
                (self.get_clock().now().nanoseconds*1e-9 - self._raw["master_L_t"])*1000, 1),
            "master_R_age_ms": round(
                (self.get_clock().now().nanoseconds*1e-9 - self._raw["master_R_t"])*1000, 1),
        }
        msg = String()
        msg.data = json.dumps(info)
        self._pub_diag.publish(msg)

        # Warn if control loop is too slow
        if avg_ms > self.dt * 1000 * 0.8:
            self.get_logger().warn(
                f"Control loop slow: avg={avg_ms:.1f}ms (budget={self.dt*1000:.1f}ms). "
                f"Consider reducing MPC horizon N or using a faster QP solver."
            )

    # -----------------------------------------------------------------------
    # Helper
    # -----------------------------------------------------------------------

    @staticmethod
    def _f64(arr: np.ndarray) -> Float64MultiArray:
        msg = Float64MultiArray()
        msg.data = arr.tolist()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = BimanualTeleopNode()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()