
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
  │  ┌──────────────────┐     ┌──────────────────┐              │
  │  │  Left MISA+MPC   │     │  Right MISA+MPC  │              │
  │  │  Controller      │     │  Controller      │              │
  │  └──────────────────┘     └──────────────────┘              │
  │         │                         │                         │
  │   MasterL ↔ SlaveL          MasterR ↔ SlaveR                │
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
  │    /teleop/diagnostics         (std_msgs/String)  [10 Hz]   │
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
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Bool, String
import json
import numpy as np
# from arm_interface import *
from .arm_interface import (
    VIPERX300S_DQ_MAX,VIPERX300S_KT,VIPERX300S_Q_MAX,VIPERX300S_Q_MIN,VIPERX300S_TAU_MAX,
    REACTORX150_DQ_MAX,REACTORX150_KT,REACTORX150_Q_MAX,REACTORX150_Q_MIN,REACTORX150_TAU_MAX,
    ArmDynamics, ArmInterface, ArmState, RX200_GRIPPER_CLOSED, RX200_GRIPPER_OPEN, VX300S_GRIPPER_CLOSED, VX300S_GRIPPER_OPEN, MAX_CURRENT_MA
)
from .misa_mpc_controller import MISAController,MISAParams
from rclpy.callback_groups import ReentrantCallbackGroup
import time
from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand



class controller(Node):

    def __init__(self):
        super().__init__('controller')
        self.get_logger().info('Controller node has started')

        # ROS2 PARAMETERS
        self.declare_parameter("control_mode","MISA")
        self.declare_parameter("loop_rate_hz",  50.0)
        self.declare_parameter("mpc_horizon",   3)
        self.declare_parameter("state_timeout_s", 0.5)


        json_data=None
        with open(r'/home/sp/Desktop/6th_Semester/RnD/ws/src/controller/controller/data.json','r') as f:
            json_data = json.load(f)
        
        self.n_joints = 8


        
        self.Ms = np.array(json_data['vx300s']['M'])
        self.Mm = np.array(json_data['rx200']['M'])

        self.Cs = np.array(json_data['vx300s']['C'])
        self.Cm = np.array(json_data['rx200']['C'])
        


        # Impedance gains (master) — diagonal entries per joint
        self.declare_parameter("master_Mmd_diag",self.Mm.flatten().tolist())
        # self.get_logger().warn(f"TRING NOW")
        # self.get_logger().warn(f"TRING TO ACCESS THE MASTER MMD PARAMETER {self.get_parameter('master_Mmd_diag').value}")


        self.declare_parameter("master_Dmd_diag", [1.0, 2.2, 1.2, 0.3, 0.2, 0.1, 0.1, 0.1])
        self.declare_parameter("master_Kmd_diag", [25.0, 60.0, 30.0, 12.0, 6.0, 3.0, 3.0, 3.0])

        # Admittance gains (slave)
        self.declare_parameter("slave_Msd_diag", self.Ms.flatten().tolist())
        self.declare_parameter("slave_Dsd_diag", [2.0, 3.0, 1.5, 0.6, 0.4, 0.2, 0.2, 0.2])
        self.declare_parameter("slave_Ksd_diag", [50.0, 80.0, 40.0, 30.0, 10.0, 5.0, 5.0, 5.0])

        # Four-channel coupling gains
        self.declare_parameter("kof", 0.8)   # slave→master force coupling
        self.declare_parameter("ksf", 0.2)   # local master force coupling
        self.declare_parameter("kom", 0.8)   # slave→master position coupling
        self.declare_parameter("ksm", 0.2)   # local master position coupling


        self.control_mode = self.get_parameter("control_mode").value
        self.frequency = self.get_parameter("loop_rate_hz").value
        self.dt = 1.0/self.frequency
        self.timeout = self.get_parameter("state_timeout_s").value

        self.qm = np.zeros(5)
        self.qs = np.zeros(5)

        self.qdotm = np.zeros(5)
        self.qdots = np.zeros(5)


        self.master_L_dynamics = ArmDynamics("reactorx150")
        self.master_R_dynamics = ArmDynamics("reactorx150")

        self.slave_L_dynamics = ArmDynamics("viperx300s")
        self.slave_R_dynamics = ArmDynamics("viperx300s")

        filter_alpha = 0.15 # ideal 0.5 - 0.8 for smoother motion
        self.master_L_interface = ArmInterface("reactorx150",self.master_L_dynamics,self.dt,filter_alpha)
        self.master_R_interface = ArmInterface("reactorx150",self.master_R_dynamics,self.dt,filter_alpha)
        self.slave_L_interface = ArmInterface("viperx300s",self.slave_L_dynamics,self.dt,filter_alpha)
        self.slave_R_interface = ArmInterface("viperx300s",self.slave_R_dynamics,self.dt,filter_alpha)


        params = self._build_params()
        self.ctrl_kwargs = dict(
            params=params,
            q_master_min=REACTORX150_Q_MIN,
            q_master_max=REACTORX150_Q_MAX,
            q_slave_min=VIPERX300S_Q_MIN,
            q_slave_max=VIPERX300S_Q_MAX,
            tau_master_max=REACTORX150_TAU_MAX,
            tau_slave_max=VIPERX300S_TAU_MAX,
            dq_master_max=REACTORX150_DQ_MAX,
            dq_slave_max=VIPERX300S_DQ_MAX,
        )

        self.ctrl_L = MISAController(
            master_dynamics=self.master_L_dynamics,
            slave_dynamics=self.slave_L_dynamics,
            **self.ctrl_kwargs
        )

        self.ctrl_R = MISAController(
            master_dynamics=self.master_R_dynamics,
            slave_dynamics=self.slave_R_dynamics,
            **self.ctrl_kwargs,
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
        self.enabled = True


        # JOINTS
        self.master_joint_names = ['waist','shoulder','elbow','wrist_angle','wrist_rotate']
        self.slaves_joint_names = ['waist','shoulder','elbow','forearm_roll','wrist_angle','wrist_rotate']
        self.n_slave_arm  = len(self.slaves_joint_names)   # 6
        self.n_master_arm = len(self.master_joint_names)  # 5

        # Gripper joint names — from grippers section of each config
        self.master_R_gripper_name = 'gripper'   # rx150 gripper
        self.slave_R_gripper_name  = 'gripper'   # vx300s gripper


        # Gripper index in raw topic array:
        # rx200:  [waist(0), shoulder(1), elbow(2), wrist_angle(3), wrist_rotate(4), gripper(5), left_finger(6), right_finger(7)]
        # vx300s: [waist(0), shoulder(1), elbow(2), forearm_roll(3), wrist_angle(4), wrist_rotate(5), gripper(6), left_finger(7), right_finger(8)]
        self.master_gripper_idx = 5   # gripper position in rx200 raw topic
        self.slave_gripper_idx  = 6   # gripper position in vx300s raw topic

        # self.default_viper_forearm_roll_angle = 180.7 * np.pi/180
        self.default_viper_forearm_roll_angle = 0.0

        self._master_gripper_pos = 0.0 

        self.master_joint_corrections = {
        # joint_name: (scale, offset_rad)
        'waist':       ( 1.0,  0.0),   # <<USER: verify — likely same convention>>
        'shoulder':    (1.0,  -3.11),   # CONFIRMED: rx200 and vx300s are opposite
        'elbow':       ( 1.0,  0.0),   # <<USER: verify>>
        'wrist_angle': ( 1.0,  0.0),   # <<USER: verify>>
        'wrist_rotate':( 1.0,  0.0),   # <<USER: verify>>
        }

        # vx300s slave corrections (usually identity, but verify)
        self.slave_joint_corrections = {
            'waist':        ( 1.0, 0.0),
            'shoulder':     ( 1.0, 0.0),
            'elbow':        ( 1.0, 0.0),
            'forearm_roll': ( 1.0, 0.0),
            'wrist_angle':  ( 1.0, 0.0),
            'wrist_rotate': ( 1.0, 0.0),
        }

        self.master_R_state = None
        self.slave_R_state = None




        #SUBSCRIPTIONS

        cbg = ReentrantCallbackGroup()

        
        self.master_state_subscription_L = self.create_subscription(
            JointState,
            'controller/master_state_L',
            lambda msg:self.master_state_callback(msg,"L"),
            10,
            callback_group=cbg
        )

        self.master_state_subscription_R = self.create_subscription(
            JointState,
            'controller/master_state_R',
            lambda msg:self.master_state_callback(msg,"R"),
            10,
            callback_group=cbg
        )

        self.slave_state_subscription_L = self.create_subscription(
            JointState,
            'controller/slave_state_L',
            lambda msg:self.slave_state_callback(msg,"L"),
            10,
            callback_group=cbg
        )

        self.slave_state_subscription_R = self.create_subscription(
            JointState,
            'controller/slave_state_R',
            lambda msg:self.slave_state_callback(msg,"R"),
            10,
            callback_group=cbg
        )

        self.master_gripper_subscriber = self.create_subscription(
            JointState,
            '/masterR/joint_states',    # direct from InterbotiX driver, unfiltered
            self._master_gripper_cb,
            10,
            callback_group=cbg
        )

        # self.final_states_slave_R = self.create_subscription(
        #     Float64MultiArray,
        #     '/controller/final_state_slaveR',
        #     lambda m: self.final_state_publisher(m,"R"),
        #     10
        # )
        # self.final_states_slave_L = self.create_subscription(
        #     Float64MultiArray,
        #     '/controller/final_state_slaveL',
        #     lambda m: self.final_state_publisher(m,"L"),
        #     10
        # )

        self.create_subscription(Bool, "/teleop/enable",
            self._enable_cb, 10, callback_group=cbg)


        # PUBLISHERS

        # self.tau_master_L_pub = self.create_publisher(
        #     Float64MultiArray,
        #     'controller/torque_master_L',
        #     10
        # )
        # self.tau_master_R_pub = self.create_publisher(
        #     Float64MultiArray,
        #     'controller/torque_master_R',
        #     10
        # )
        # self.q_slave_L_pub = self.create_publisher(
        #     Float64MultiArray,
        #     'controller/final_state_slaveL',
        #     10
        # )
        # self.q_slave_R_pub = self.create_publisher(
        #     Float64MultiArray,
        #     'controller/final_state_slaveR',
        #     10
        # )

        # self.q_retrieved_R_pub = self.create_publisher(
        #     JointState,
        #     'slaveR/joint_states',
        #     10
        # )

        # Slave arm position commands — mirrors xsarm_puppet_single.cpp exactly
        self.slave_R_arm_pub = self.create_publisher(
            JointGroupCommand,
            '/slaveR/commands/joint_group',   # InterbotiX standard topic
            10
        )
        self.slave_R_gripper_pub = self.create_publisher(
            JointSingleCommand,
            '/slaveR/commands/joint_single',
            10
        )
        self.master_R_torque_pub = self.create_publisher(
            JointGroupCommand,
            '/masterR/commands/joint_group',
            10
        )
        # NEED TO CREATE LEFT PUBLISHER AS WELL

        self.diagnostics_pub = self.create_publisher(
            String,
            'controller/teleop/diagnostics',
            10
        )

        self.master_R_gripper_torque_pub = self.create_publisher(
            JointSingleCommand,
            '/masterR/commands/joint_single',
            10
        )


        self.control_timer = self.create_timer(self.dt, self.control_loop,callback_group=cbg)

        self.diagnostics_timer = self.create_timer(0.1, self.publish_diagnostics,callback_group=cbg)


        self._loop_times = []


        self.get_logger().info(f"""
Controller Node initialized
Control Mode : {self.control_mode}
dt : {self.dt*1000:.1f} ms
        """)

    def _extract_and_correct(
        self,
        msg: JointState,
        joint_names: list,
        corrections: dict,
        n_total: int = 8
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract joints by name, apply sign/offset corrections, pad to n_total.
        Returns (q_corrected, effort) both shape (n_total,)
        """
        msg_names = list(msg.name)
        has_effort = len(msg.effort) == len(msg.name)

        q_out = np.zeros(n_total)
        I_out = np.zeros(n_total)

        for i, name in enumerate(joint_names):
            if name not in msg_names:
                self.get_logger().warn(
                    f"Joint '{name}' not in message. Available: {msg_names}",
                    throttle_duration_sec=5.0
                )
                continue
            idx = msg_names.index(name)
            scale, offset = corrections.get(name, (1.0, 0.0))
            q_out[i] = scale * msg.position[idx] + offset
            I_out[i] = msg.effort[idx] if has_effort else 0.0

        return q_out, I_out

    def _master_gripper_cb(self, msg: JointState):
        """Extract gripper position from unfiltered master joint states."""
        msg_names = list(msg.name)
        if 'gripper' in msg_names:
            idx = msg_names.index('gripper')
            self._master_gripper_pos = float(msg.position[idx])
            self.get_logger().debug(
                f"Master gripper raw={self._master_gripper_pos:.4f}",
                throttle_duration_sec=1.0
            )

    def _enable_cb(self,msg):
        self.enabled = msg.data
        state = "enabled" if self.enabled else "disabled"
        self.get_logger().info(f"Teleoperation {state} via /teleop/enable topic")

    def publish_diagnostics(self):
        if not self._loop_times:
            return
        
        avg_ms = np.mean(self._loop_times)*1000
        max_ms = np.max(self._loop_times)*1000

        info = {
            "enabled" : self.enabled,
            "control_mode" : self.control_mode,
            "Loop average ms " : avg_ms,
            "Loop max ms" : max_ms,
            "master_L_age_ms": round(
                (self.get_clock().now().nanoseconds*1e-9 - self._raw["master_L_t"])*1000, 1),
            "master_R_age_ms": round(
                (self.get_clock().now().nanoseconds*1e-9 - self._raw["master_R_t"])*1000, 1),
        }

        msg=String()
        msg.data = json.dumps(info)
        self.diagnostics_pub.publish(msg)

    

    def _build_params(self) -> MISAParams:
        get = lambda name: np.array(self.get_parameter(name).value, dtype=float)
 
        return MISAParams(
            n_joints = 8,
            N  = self.get_parameter("mpc_horizon").value,
            dt = self.dt,
            Mmd = np.array(get("master_Mmd_diag")).reshape(8,8),
            Dmd = np.diag(get("master_Dmd_diag")),
            Kmd = np.diag(get("master_Kmd_diag")),
            Msd = np.array(get("slave_Msd_diag")).reshape(8,8),
            Dsd = np.diag(get("slave_Dsd_diag")),
            Ksd = np.diag(get("slave_Ksd_diag")),
            kof = float(self.get_parameter("kof").value),
            ksf = float(self.get_parameter("ksf").value),
            kom = float(self.get_parameter("kom").value),
            ksm = float(self.get_parameter("ksm").value),
        )

    def control_loop(self):
        if not self.enabled:
            self.get_logger().warn("Teleoperation Loop is disabled.")
            return

        t_start = time.perf_counter()
        now = self.get_clock().now().nanoseconds * 1e-9


        # Check for data freshness and ignore if any arms state is too old

        #HAVE TO CHANGEEEEEE
        arms_list = ["master_R", "slave_R"]
        for arm_key in arms_list:
            age = now - self._raw.get(f"{arm_key}_t", 0.0)
            if age > self.timeout:
                self.get_logger().warn(
                    f"[{arm_key}] stale state ({age*1000:.0f}ms) — skipping cycle")
                return

        # Check all data present
        for arm_key in arms_list:
            if (self._raw.get(f"{arm_key}_q") is None or
                    self._raw.get(f"{arm_key}_I") is None):
                self.get_logger().info(f"Waiting for {arm_key} data...")
                return

        # Process arm states 

        # master_L_state = self.master_L_interface.update(
        #     self._raw['master_L_q'], self._raw['master_L_I'])
        
        master_R_state = self.master_R_interface.update(
            self._raw['master_R_q'], self._raw['master_R_I'])
        
        # slave_L_state = self.slave_L_interface.update(
        #     self._raw['slave_L_q'],self._raw['slave_L_I']
        # )
        slave_R_state = self.slave_R_interface.update(
            self._raw['slave_R_q'],self._raw['slave_R_I']
        )

        self.master_R_state = master_R_state
        self.slave_R_state = slave_R_state

        self.get_logger().info(
                f"master_q={np.round(master_R_state.q[:6], 3)} "
                f"slave_q={np.round(slave_R_state.q[:6], 3)}",
                throttle_duration_sec=1.0
            )
    
        self.get_logger().info(
            f"\nmaster: w={master_R_state.q[0]:.2f} sh={master_R_state.q[1]:.2f} "
            f"el={master_R_state.q[2]:.2f} wa={master_R_state.q[3]:.2f} wr={master_R_state.q[4]:.2f}"
            f"\nslave:  w={slave_R_state.q[0]:.2f} sh={slave_R_state.q[1]:.2f} "
            f"el={slave_R_state.q[2]:.2f} fr={slave_R_state.q[3]:.2f} "
            f"wa={slave_R_state.q[4]:.2f} wr={slave_R_state.q[5]:.2f}",
            throttle_duration_sec=1.0
        )

        self.get_logger().info(
            f"tau_ext_master={np.round(master_R_state.tau_ext[:5], 3)} "
            f"tau_ext_slave={np.round(slave_R_state.tau_ext[:5], 3)}",
            throttle_duration_sec=1.0
        )


        # Running MPC Controller

        # Right Pair
        # tau_master_L , q_slave_left =  self.ctrl_L.step(master_L_state, slave_L_state)
        # Left Pair
        tau_master_R, q_slave_right = self.ctrl_R.step(master_R_state, slave_R_state)



        # SAFETY CLAMPING
        # tau_master_L = np.clip(tau_master_L, -self.ctrl_kwargs['tau_master_max'], self.ctrl_kwargs['tau_master_max'])
        tau_master_R = np.clip(tau_master_R, -self.ctrl_kwargs['tau_master_max'], self.ctrl_kwargs['tau_master_max'])

        # q_slave_left = np.clip(q_slave_left, self.ctrl_kwargs['q_slave_min'], self.ctrl_kwargs['q_slave_max'])
        q_slave_right = np.clip(q_slave_right, self.ctrl_kwargs['q_slave_min'], self.ctrl_kwargs['q_slave_max'])


        # PUBLISHING RESULTS FROM THE CONTROL LOOP

        

        self.get_logger().fatal(f"PUBLISHING MESSAGES FOR TORQUE NOW!")
        self.get_logger().warn(f'Torque being pushed = {tau_master_R}\nJoint Angles slave = {q_slave_right}')
        # self.tau_master_L_pub.publish(self._f64(tau_master_L))
        # self.tau_master_R_pub.publish(self._f64(tau_master_R))

        # # self.q_slave_L_pub.publish(self._f64(q_slave_left))
        # self.q_slave_R_pub.publish(self._f64(q_slave_right))
        self._publish_slave_command(q_slave_right, side="R")
        self._publish_master_torque(tau_master_R, side="R")

        self.get_logger().info(
            f"slave_cmd={np.round(q_slave_right[:self.n_slave_arm], 3)} "
            f"tau_master={np.round(tau_master_R[:self.n_master_arm], 3)}",
            throttle_duration_sec=0.5   # log at 2Hz, not 50Hz
        )
                



        self._loop_times.append(time.perf_counter() - t_start)
        if(len(self._loop_times)>100):
            self._loop_times.pop(0)
        
        return
    

    # def _publish_slave_command(self, q_slave: np.ndarray, side: str = "R"):
    #     """
    #     Send joint position commands to the slave arm.
    #     Mirrors xsarm_puppet_single.cpp:
    #     pub_group  -> follower/commands/joint_group  (arm joints)
    #     pub_single -> follower/commands/joint_single (gripper)
        
    #     q_slave: full 8-DOF array from MPC output
    #     """
    #     # --- Arm group command (first n_slave_arm joints) ---
    #     arm_msg = JointGroupCommand()
    #     arm_msg.name = "arm"
    #     # Take only the arm joints, not gripper/finger joints
    #     q_slave = list(q_slave)
    #     q_slave.insert(3,self.default_viper_forearm_roll_angle)

    #     arm_cmd = q_slave[:self.n_slave_arm]

    #     # Sanity check before sending
    #     if not np.all(np.isfinite(arm_cmd)):
    #         self.get_logger().warn(
    #             f"[slave_{side}] NaN/inf in arm command, holding position",
    #             throttle_duration_sec=1.0
    #         )
    #         # Hold at last known good slave position
    #         if self._raw.get(f'slave_{side}_q') is not None:
    #             arm_cmd = self._raw[f'slave_{side}_q'][:self.n_slave_arm]
    #         else:
    #             return
    #     arm_cmd = np.array(arm_cmd)
    #     arm_msg.cmd = arm_cmd.tolist()

    #     # --- Gripper command ---
    #     # Pass master gripper position through directly to slave gripper.
    #     # The *2 scaling matches xsarm_puppet_single.cpp exactly —
    #     # gripper joint pos = half the total finger spread in InterbotiX convention.
    #     # <<USER: If your gripper behaves differently, adjust this scaling.>>
    #     gripper_msg = JointSingleCommand()
    #     gripper_msg.name = "gripper"

    #     master_q = self._raw.get(f'master_{side}_q')
    #     if master_q is not None and len(master_q) > 5:
    #         # Index 5 = gripper joint in rx150 joint ordering
    #         gripper_cmd = float(master_q[5]) * 2.0
    #     else:
    #         gripper_cmd = 0.0   # neutral position

    #     gripper_msg.cmd = gripper_cmd

    #     if side == "R":
    #         self.slave_R_arm_pub.publish(arm_msg)
    #         self.slave_R_gripper_pub.publish(gripper_msg)
    #         self.get_logger().debug(
    #             f"Slave R arm cmd: {np.round(arm_cmd, 3)}, gripper: {gripper_cmd:.3f}"
    #         )
    #     # <<USER: Add side == "L" block when you wire up the left pair>>

    def _publish_slave_command(self, q_slave: np.ndarray, side: str = "R"):
        
        arm_cmd = np.array(q_slave[:(self.n_slave_arm-1)])
        
        # Insert forearm_roll at index 3 with CORRECT home angle (0 rad)
        # <<USER: Only change this offset if your physical mounting requires it>>
        if self.n_slave_arm == 6:
            # vx300s: MPC gives [waist,shoulder,elbow,wrist_angle,wrist_rotate,...]
            # but vx300s arm group needs [waist,shoulder,elbow,forearm_roll,wrist_angle,wrist_rotate]
            arm_cmd_6 = np.insert(arm_cmd[:5], 3, 0.0)   # insert 0.0 for forearm_roll
        else:
            arm_cmd_6 = arm_cmd

        # --- Rate limiter: cap per-cycle joint movement ---
        # <<USER: MAX_DELTA_PER_CYCLE = max radians per control cycle (1/50 Hz = 20ms)
        #         0.002 rad/cycle at 50Hz = 0.1 rad/s — very conservative
        #         Increase to 0.01 once behaviour is confirmed correct>>
        MAX_DELTA_PER_CYCLE = 0.12  # radians per 20ms cycle

        if not hasattr(self, f'_last_slave_{side}_cmd'):
            # First cycle: initialise from current slave position
            current_q = self._raw.get(f'slave_{side}_q')
            current_q = np.insert(current_q[:(self.n_slave_arm-1)], 3, 0.0)
            if current_q is not None:
                setattr(self, f'_last_slave_{side}_cmd', current_q)
            else:
                setattr(self, f'_last_slave_{side}_cmd', arm_cmd_6.copy())

        last_cmd = getattr(self, f'_last_slave_{side}_cmd')
        delta = arm_cmd_6 - last_cmd
        delta_clipped = np.clip(delta, -MAX_DELTA_PER_CYCLE, MAX_DELTA_PER_CYCLE)
        arm_cmd_6 = last_cmd + delta_clipped
        setattr(self, f'_last_slave_{side}_cmd', arm_cmd_6.copy())

        # Final sanity check
        if not np.all(np.isfinite(arm_cmd_6)):
            self.get_logger().warn(f"[slave_{side}] NaN after rate limiter — holding")
            arm_cmd_6 = last_cmd

        arm_msg = JointGroupCommand()
        arm_msg.name = "arm"
        arm_msg.cmd = arm_cmd_6.tolist()   # exactly 6 floats for vx300s

        # Gripper passthrough
        # master_q = self._raw.get(f'master_{side}_q')
        # gripper_cmd = float(master_q[5]) * 2.0 if (
        #     master_q is not None and len(master_q) > 5
        # ) else 0.0

        # The *2 scaling is from xsarm_puppet_single.cpp — gripper joint stores
        # half the total finger spread in InterbotiX convention
        gripper_raw = float(getattr(self, '_master_gripper_pos', 0.0)) 
        gripper_cmd = float(np.interp(
            gripper_raw,
            [RX200_GRIPPER_CLOSED, RX200_GRIPPER_OPEN],
            [VX300S_GRIPPER_CLOSED, VX300S_GRIPPER_OPEN]
        ))
        self.get_logger().fatal(f'GRIPPER POSITION = {gripper_cmd}')

        # Clamp to valid gripper range for vx300s
        # <<USER: verify these limits from vx300s gripper spec>>
        gripper_cmd = float(np.clip(gripper_cmd, -0.037, 0.037))

        gripper_msg = JointSingleCommand()
        gripper_msg.name = "gripper"
        gripper_msg.cmd = gripper_cmd


        self.get_logger().info(
            f"gripper raw={gripper_raw:.4f} → cmd={gripper_cmd:.5f}",
            throttle_duration_sec=0.5
        )

        

        if side == "R":
            self.slave_R_arm_pub.publish(arm_msg)
            self.slave_R_gripper_pub.publish(gripper_msg)
            self.get_logger().info(
                f"gripper_cmd={gripper_cmd:.4f} from master_pos={getattr(self, '_master_gripper_pos', 0.0):.4f}",
                throttle_duration_sec=1.0
                )   

    def _publish_master_torque(self, tau_master: np.ndarray, side: str = "R"):
        """
        Send torque feedback commands to the master arm.
        
        <<USER: This only has physical effect if masterR is in
                Current-Based Position Control mode (Operating_Mode: 5).
                In pure position mode the effort field is ignored by the driver.
                
                To enable current mode on masterR, in your motor config YAML:
                all:
                    Operating_Mode: 5
                    Profile_Velocity: 0
                    Profile_Acceleration: 0
                Then relaunch the arm driver.>>
        """

        tau_cmd = tau_master[:self.n_master_arm]   # shape (5,) for rx150

        if not np.all(np.isfinite(tau_cmd)):
            self.get_logger().warn(f"[master_{side}] NaN in torque — sending zero current")
            tau_cmd = np.zeros(self.n_master_arm)
            # return

        # Convert torque to current: I = tau / kt
        kt = REACTORX150_KT[:self.n_master_arm]   # [Nm/A] per joint
        current_cmd = tau_cmd / kt                  # [A] per joint
        current_mA = current_cmd*1000

        # For haptic feedback, use 20-30% of max rated current
        # This gives gentle but perceptible feedback without violent motion
        HAPTIC_CURRENT_LIMIT_MA = np.array([
            100.0,   # waist — 400mA ≈ 15% of XM430 max
            150.0,   # shoulder — 600mA (dual motor, so effective torque is doubled)
            100.0,   # elbow
            50.0,   # wrist_angle
            50.0,   # wrist_rotate
        ])
        # <<USER: Start at these conservative values.
        #         Increase by 100mA increments until feedback feels natural.
        #         The arm should resist contact but not overpower the operator.>>
        filter = 0.5
        current_mA = np.clip(current_mA, filter*(-HAPTIC_CURRENT_LIMIT_MA), filter*(HAPTIC_CURRENT_LIMIT_MA))


        # current_mA = np.clip(current_mA,-MAX_CURRENT_MA, MAX_CURRENT_MA)


        torque_msg = JointGroupCommand()
        torque_msg.name = "arm"
        # Send only the arm joints worth of torque values
        torque_msg.cmd =current_mA.tolist()
        self.get_logger().fatal(
        f"[master_{side}] tau={np.round(tau_cmd, 3)} → I={np.round(current_mA, 1)} mA",
        )

        gripper_tau = tau_master[5] if len(tau_master) > 5 else 0.0
        gripper_kt  = REACTORX150_KT[5]
        gripper_mA  = float(np.clip((gripper_tau / gripper_kt) * 1000.0, -500.0, 500.0))

        gripper_torque_msg = JointSingleCommand()
        gripper_torque_msg.name = "gripper"
        gripper_torque_msg.cmd = gripper_mA

        if side == "R":
            self.master_R_gripper_torque_pub.publish(gripper_torque_msg)

        if side == "R":
            self.master_R_torque_pub.publish(torque_msg)

        # Gripper force feedback:
        # When slave gripper contacts an object, tau_ext_slave[6] becomes non-zero
        # This is reflected back to make master gripper harder to close
        # gripper_total_tau = 
        # gripper_kt  = REACTORX150_KT[5]   # XL430 kt
        # gripper_mA  = float(np.clip(
        #     (gripper_total_tau / gripper_kt) * 1000.0,
        #     -500.0, 500.0   # 500mA max for XL430 gripper
        # ))


        

    def master_state_callback(self,msg:JointState,arm:str):
        
        arm = "master_L" if "L" == arm else "master_R"
        # q_raw = np.array(msg.position)
        # I_raw = np.array(msg.effort)
        q_raw,I_raw = self._extract_and_correct(
            msg=msg,
            joint_names=self.master_joint_names,
            corrections=self.master_joint_corrections
        )

        # msg_names = list(msg.name)
        # # msg_names = list(msg.name)
        # if 'gripper' in msg_names:
        #     gripper_idx = msg_names.index('gripper')
        #     self._master_gripper_pos = float(msg.position[gripper_idx])
        # else:
        #     # Fallback: gripper is at index 5 in rx200 raw array
        #     # <<USER: verify this index matches your actual topic output>>
        #     if len(msg.position) > 5:
        #         self._master_gripper_pos = float(msg.position[5])

        t = self.get_clock().now().nanoseconds * 1e-9
        self._raw[f"{arm}_q"] = q_raw
        self._raw[f"{arm}_I"] = I_raw
        self._raw[f"{arm}_t"] = t

    def slave_state_callback(self,msg:JointState,arm:str):
        arm = "slave_L" if "L" == arm else "slave_R"

        q_raw,I_raw = self._extract_and_correct(
            msg=msg,
            joint_names=self.master_joint_names,
            corrections=self.slave_joint_corrections
        )

        # msg_names = list(msg.name)
        # if 'gripper' in msg_names:
        #     self._master_gripper_pos = msg.position[msg_names.index('gripper')]
            
        t = self.get_clock().now().nanoseconds * 1e-9
        self._raw[f"{arm}_q"] = q_raw
        self._raw[f"{arm}_I"] = I_raw
        self._raw[f"{arm}_t"] = t
    
    def final_state_publisher(self,msg: Float64MultiArray, s:str):
        joints = msg
        
        output = JointState()
        output.name=self.slaves_joint_names
        output.position = list(msg.data)
        
        self.q_retrieved_R_pub.publish(output)

    
    @staticmethod
    def _f64(arr: np.ndarray)->Float64MultiArray:
        msg = Float64MultiArray()
        msg.data = arr.tolist()
        return msg

        



def main(args=None):
    rclpy.init(args=args)
    node = controller()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()

