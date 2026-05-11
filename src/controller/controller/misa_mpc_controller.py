"""
misa_mpc_controller.py
-----------------------
Master Impedance - Slave Admittance (MISA) bilateral controller
with MPC-based optimal trajectory generation.
 
Control Philosophy
------------------------------------------------------
 
MASTER (ViperX300s) — Impedance Control
  The operator physically moves the master arm in current-control mode.
  The impedance law renders a virtual mechanical environment to the
  operator and incorporates slave feedback forces:
 
    tau_m = M_m * ddq_m + C_m + G_m - J_m^T * F_h
          = tau_m_cmd + delta_tau_s_p + delta_tau_s_f
 
  where:
    tau_m_cmd  = motor commands for master's own smooth motion
    delta_tau_s_p = torque from position tracking error (master vs slave)
    delta_tau_s_f = torque from slave's external contact force feedback
 
SLAVE (ReactorX150) — Admittance Control
  The slave receives the master's joint angles as a position reference.
  It complies with environment contact forces via:
 
    M_sd * ddq_e + D_sd * dq_e + K_sd * q_e = J_s^T * (F_e - F_m_f - F_m_p)
 
  where F_e is estimated from motor current residuals (no F/T sensor).
 
MPC Integration
---------------
  At each timestep the MPC solves for the optimal torque/position
  trajectory over a finite horizon N, subject to:
    - Joint position / velocity / torque limits
    - Dynamics propagation model
    - Tracking the reference q_ref (from master→slave or operator intent)
 
  The first element of the optimal sequence is applied (receding horizon).
 
Four-Channel Teleoperation Channels
-------------------------------------
  Ch1: master position → slave reference       (position feed-forward)
  Ch2: slave position  → master feedback       (position error term)
  Ch3: slave contact force → master torque     (force feed-forward)
  Ch4: master operator force → slave reference (force feed-forward)
 
Sensor Constraints
------------------
  No external F/T sensors.  All forces are estimated from:
    tau_ext = tau_motor - (C + G)    [M*ddq omitted — too noisy]
  and mapped to Cartesian space via the Jacobian transpose:
    F_est = (J^T)^+ * tau_ext
 
"""



import numpy as np
from dataclasses import dataclass
from typing import Tuple
import json
from .mpc_optimizer import MPCOptimizer
from .arm_interface import ArmState, ArmDynamics
import pinocchio as pin

import numpy.linalg as la
from .arm_interface import VIPERX300S_TAU_MAX, VIPERX300S_DQ_MAX


# ---------------------------------------------------------------------------
# Controller parameters (overridden from YAML in the ROS2 node)
# ---------------------------------------------------------------------------
@dataclass
class MISAParams:
    # n_joints: int = 5
    # Joints are considered to be 8
    n_joints:int = 8
 
    # Impedance law gains (master side) — joint space
    # <<USER: Tune these for your operator's preferred feel.
    #         Higher Kmd = stiffer tracking of slave position error.
    #         Higher Dmd = more damping (smoother, less oscillation).>>
    Mmd: np.ndarray = None   # Virtual inertia  (5x5 diagonal)
    Dmd: np.ndarray = None   # Virtual damping  (5x5 diagonal)
    Kmd: np.ndarray = None   # Virtual stiffness(5x5 diagonal)
 
    # Admittance law gains (slave side) — joint space
    # <<USER: Tune these for compliant interaction with environment.
    #         Lower Ksd = more compliant to external forces.>>
    Msd: np.ndarray = None
    Dsd: np.ndarray = None
    Ksd: np.ndarray = None
 
    # Force feedback coupling gains (four-channel architecture)
    # kof: remote torque feedback weight (slave → master)
    # ksf: local torque feedback weight  (master self-torque)
    # kom: remote motion feedback weight (slave position → master)
    # ksm: local motion feedback weight  (master tracking)
    kof: float = 0.8   # <<USER: tune in [0,1]; 1 = full slave force feedback>>
    ksf: float = 0.2   # <<USER: tune in [0,1]>>
    kom: float = 0.8   # <<USER: tune in [0,1]>>
    ksm: float = 0.2   # <<USER: tune in [0,1]>>
 
    # MPC horizon and sample time
    N:  int   = 3
    dt: float = 0.02   # 100 Hz control loop


    



    def __post_init__(self):
        n = self.n_joints
        if self.Mmd is None:
            # <<USER: Replace diagonal values with physically meaningful inertia>>
            self.Mmd = np.diag([1.0, 1.5, 1.0, 0.5, 0.3])
        if self.Dmd is None:
            # <<USER: Set damping to suppress oscillations from force feedback>>
            self.Dmd = np.diag([5.0, 8.0, 5.0, 3.0, 2.0, 0.5, 0.5, 0.5])
        if self.Kmd is None:
            # <<USER: Stiffness couples master to slave position error>>
            self.Kmd = np.diag([10.0, 15.0, 10.0, 5.0, 3.0, 1.0, 1.0, 1.0])
        if self.Msd is None:
            self.Msd = np.diag([0.8, 1.2, 0.8, 0.4, 0.25, 0.1, 0.1, 0.1])
        if self.Dsd is None:
            # <<USER: Higher damping = slower but smoother slave response>>
            self.Dsd = np.diag([4.0, 6.0, 4.0, 2.5, 1.5, 0.5, 0.5, 0.5])
        if self.Ksd is None:
            # <<USER: Lower stiffness = more compliance with env. contact>>
            self.Ksd = np.diag([8.0, 12.0, 8.0, 4.0, 2.5, 0.5, 0.5, 0.5])

def _safe_inv(M: np.ndarray, reg: float = 1e-4) -> np.ndarray:
    """
    Tikhonov-regularised matrix inverse.
    Adds reg*I before inverting to prevent blow-up on near-singular matrices.
    
    <<USER: reg=1e-4 is a good starting point. Increase to 1e-3 if overflow
            persists. Too large a reg makes the dynamics inaccurate but stable.>>
    """
    n = M.shape[0]
    return np.linalg.inv(M + reg * np.eye(n))


class MISAController:
    '''
    Full MISA Controller
    '''

    def __init__(self,
                params: MISAParams,
                master_dynamics: ArmDynamics,
                slave_dynamics: ArmDynamics,
                q_master_min: np.ndarray , 
                q_master_max: np.ndarray,
                q_slave_min:  np.ndarray,
                q_slave_max: np.ndarray,

                tau_master_max: np.ndarray,
                tau_slave_max: np.ndarray,
                
                dq_master_max: np.ndarray,
                dq_slave_max:np.ndarray):
        
        self.p = params
        self.n_joints = params.n_joints

        self.master_dyn = master_dynamics
        self.slave_dyn = slave_dynamics

                # Build MPC optimizers — one per arm
        # MPC weight matrices
        # <<USER: Tune Q/R/S/P to balance tracking vs effort vs terminal cost>>
        Q_m = np.diag([70.0]*self.n_joints)   # master position tracking
        R_m = np.diag([0.3]*self.n_joints)    # master control effort
        S_m = np.diag([0.05]*self.n_joints)   # master torque smoothness
        P_m = 5 * Q_m                  # master terminal cost
 
        Q_s = np.diag([150.0]*self.n_joints)   # slave position tracking (tighter)
        R_s = np.diag([0.05]*self.n_joints)
        S_s = np.diag([0.02]*self.n_joints)
        P_s = 10 * Q_s
 
        self.master_mpc = MPCOptimizer(
            n_joints=self.n_joints, horizon=params.N, dt=params.dt,
            Q=Q_m, R=R_m, S=S_m, P=P_m,
            q_min=q_master_min, q_max=q_master_max,
            dq_min=-dq_master_max, dq_max=dq_master_max,
            tau_min=-tau_master_max, tau_max=tau_master_max,
        )
 
        self.slave_mpc = MPCOptimizer(
            n_joints=self.n_joints, horizon=params.N, dt=params.dt,
            Q=Q_s, R=R_s, S=S_s, P=P_s,
            q_min=q_slave_min, q_max=q_slave_max,
            dq_min=-dq_slave_max, dq_max=dq_slave_max,
            tau_min=-tau_slave_max, tau_max=tau_slave_max,
        )
 
        # Velocity integrator state for admittance law
        self._dq_slave_int:  np.ndarray = np.zeros(self.n_joints)
        self._q_slave_int:   np.ndarray = None    # initialised on first call
 
        # Position error integrators
        self._e_master: np.ndarray = np.zeros(self.n_joints)
        self._e_slave:  np.ndarray = np.zeros(self.n_joints)

        self._q_compliance  = np.zeros(self.n_joints)   # admittance position offset
        self._dq_compliance = np.zeros(self.n_joints)   # admittance velocity offset
        self._q_slave_int   = None  
 
    # -----------------------------------------------------------------------
    # Four-channel bilateral control step
    # -----------------------------------------------------------------------
 
    def step(
        self,
        master_state: ArmState,   # Current master arm state
        slave_state:  ArmState,   # Current slave  arm state
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute one control cycle.
 
        Returns
        -------
        tau_master_cmd : (5,) torque commands to master arm motors
        q_slave_cmd    : (5,) position commands to slave  arm motors
        """
        dt = self.p.dt
        q_m  = master_state.q
        dq_m = master_state.dq
        q_s  = slave_state.q
        dq_s = slave_state.dq
 
        # Initialise slave integrator on first call
        # Initialise integrator on first call to current slave position
        # This prevents the first command from jumping to an arbitrary location
        if self._q_slave_int is None:
            self._q_slave_int  = q_s.copy()
            self._q_compliance = np.zeros(self.n_joints)
            self._dq_compliance = np.zeros(self.n_joints)
            self.get_logger().info("Slave integrator initialised to current position") \
                if hasattr(self, '_logger') and self._logger else None
            # On the very first cycle, just return current positions — don't move yet
            # <<USER: This one-cycle delay ensures the integrator is stable before
            #         the first command is sent to the hardware>>
            return np.zeros(self.n_joints), q_s.copy()
 
        # ------------------------------------------------------------------
        # STEP 1: Get robot dynamics at current states
        # ------------------------------------------------------------------
        M_m = self.master_dyn.get_M(q_m)
        C_m = self.master_dyn.get_C(q_m, dq_m)
        G_m = self.master_dyn.get_G(q_m)
 
        M_s = self.slave_dyn.get_M(q_s)
        C_s = self.slave_dyn.get_C(q_s, dq_s)
        G_s = self.slave_dyn.get_G(q_s)


        # Temporarily in misa_mpc_controller.py step(), after computing G_m:
        print(f"q_m[:5]={np.round(q_m[:5],3)}, G_m[:5]={np.round(G_m[:5],3)}")
 
        # ------------------------------------------------------------------
        # STEP 2: Extract torque feedback signals (four-channel architecture)
        #
        # tau_ext_slave  = external contact torque estimated on slave
        # tau_ext_master = operator interaction torque estimated on master
        #
        # These are estimated from motor current residuals in arm_interface.py
        # ------------------------------------------------------------------
        tau_ext_slave  = slave_state.tau_ext    # Ch3: slave contact → master
        tau_ext_master = master_state.tau_ext   # Ch4: operator force → slave

        # ------------------------------------------------------------------
        # ADMITTANCE CONTROL (slave side) — paper Eq. 5
        #
        # Computes a compliant joint-angle offset on top of q_master.
        # The ODE governs how the slave *yields* to contact forces:
        #
        #   M_sd * ddq_e + D_sd * dq_e + K_sd * q_e
        #       = tau_ext_slave - tau_reflected_master
        #
        # where q_e is the compliance offset (NOT the tracking error).
        # We integrate this ODE to get q_e, then:
        #   q_slave_ref = q_master + q_e
        #
        # Separate integrator buffers:
        #   self._q_compliance  — integrated position offset (q_e)
        #   self._dq_compliance — integrated velocity offset (dq_e)
        # ------------------------------------------------------------------


        # Step A: validate integrator state before using it
        if (not np.all(np.isfinite(self._q_compliance)) or
                not np.all(np.isfinite(self._dq_compliance))):
            self.get_logger().warn("Compliance integrator reset") \
                if hasattr(self, '_logger') and self._logger else None
            self._q_compliance  = np.zeros(self.n)
            self._dq_compliance = np.zeros(self.n)


        # Position compliance offset (starts at zero — slave starts at master pose)
        q_e  = self._q_compliance           # current compliance offset
        dq_e = self._dq_compliance          # current compliance velocity

        
 
        # Four-channel feedback torque construction (paper Eq. 9, unified arch.)
        # Remote torque feedback to master (slave contact forces):
        #   delta_tau_s_f = -kof * tau_ext_slave
        # Remote position error feedback to master:
        #   delta_tau_s_p = -Kmd * (q_m - q_s)


        # Changed code
        # delta_tau_s_f = -self.p.kof * tau_ext_slave
        # delta_tau_s_p = -self.p.Kmd @ (q_m - q_s)
        # tau_reflected_master = self.p.kof * tau_ext_master

        # Net driving torque for admittance ODE:
        #   env contact pushes slave away from master reference (+)
        #   reflected master force resists that deviation  (-)
        # tau_admittance = tau_ext_slave - tau_reflected_master




        # Step B: validate tau signals before feeding into ODE
        tau_reflected_master = self.p.kof * tau_ext_master
        if not np.all(np.isfinite(tau_reflected_master)):
            tau_reflected_master = np.zeros(self.n)

        tau_admittance = tau_ext_slave - tau_reflected_master
        if not np.all(np.isfinite(tau_admittance)):
            tau_admittance = np.zeros(self.n)

        # Step C: clamp tau_admittance to physically reasonable range
        # Large torques from bad current sensing will otherwise blow up the ODE
        # <<USER: TAU_CLAMP should be set to your motor's max rated torque.
        #         XM430: 3.1 Nm stall, XM540: 10.0 Nm stall>>
        TAU_CLAMP = VIPERX300S_TAU_MAX
        tau_admittance = np.clip(tau_admittance, -TAU_CLAMP, TAU_CLAMP)

        # Integrate the admittance ODE (Euler, one step):
        #   M_sd * ddq_e = tau_admittance - D_sd * dq_e - K_sd * q_e
        M_sd_inv = _safe_inv(self.p.Msd)
        # M_sd_inv = la.pinv(self.p.Msd)
        cond = la.cond(self.p.Msd)
        if cond > 1e6:
            # <<USER: If this fires, your Msd is nearly singular.
            #         Use a regularised inverse instead.>>
            print('='*100)
            print(f"WARNING: Msd condition number = {cond:.1e} — nearly singular")
            print('='*100)

        # Step D: compute ODE terms individually, check each before combining
        Dsd_dq = self.p.Dsd @ dq_e
        Ksd_qe = self.p.Ksd @ q_e

        if not np.all(np.isfinite(Dsd_dq)):
            Dsd_dq = np.zeros(self.n)
        if not np.all(np.isfinite(Ksd_qe)):
            Ksd_qe = np.zeros(self.n)

        ddq_e = M_sd_inv @ (
            tau_admittance
            - Dsd_dq     # velocity damping
            - Ksd_qe       # spring restoring force toward q_master
        )


        # Step F: clamp ddq_e before integrating — this is the critical guard
        # Without this, a single bad M_sd_inv can produce ddq_e = 1e15
        # and the integrator overflows in one step
        # <<USER: 20 rad/s^2 is conservative. Your motors physically cannot
        #         accelerate faster than ~50 rad/s^2 at rated current.>>

        MAX_DDQ = 30.0
        if not np.all(np.isfinite(ddq_e)) or np.any(np.abs(ddq_e) > MAX_DDQ):
            ddq_e = np.clip(
                np.where(np.isfinite(ddq_e), ddq_e, 0.0),
                -MAX_DDQ, MAX_DDQ
            )
        
        # Update compliance integrators
        self._dq_compliance = dq_e + self.p.dt * ddq_e
        self._q_compliance  = q_e  + self.p.dt * self._dq_compliance

        # Compliance offset must stay physically small — arm can't deviate far from master
        # <<USER: 0.15 rad = ~8.5 degrees max compliance deviation. Tighten for precision.>>
        MAX_COMPLIANCE = 0.05
        self._q_compliance = np.clip(self._q_compliance, -MAX_COMPLIANCE, MAX_COMPLIANCE)

        # Slave position reference = master joints + compliant offset
        # When tau_ext_slave = 0 (no contact): q_compliance → 0,
        # so slave simply mirrors master
        # When contact occurs: slave yields compliantly

        # Clamp compliance velocity too
        MAX_DQ_COMPLIANCE = 0.5   # rad/s
        self._dq_compliance = np.clip(self._dq_compliance, -MAX_DQ_COMPLIANCE, MAX_DQ_COMPLIANCE)
        q_slave_ref = q_m + self._q_compliance


        # REMOVED
        # # Slave admittance: force terms from master (remote)
        # #   F_m_f: master operator torque reflected to slave
        # #   F_m_p: position error term in slave admittance law
        # F_m_f = self.p.ksf * tau_ext_master
        # F_m_p = self.p.ksm * (q_s - q_m)   # drives slave toward master
 
        # ------------------------------------------------------------------
        # STEP 3: Master Impedance Law (paper Eq. 9)
        #
        # tau_k^m = M_m * ddq_m + C_m + G_m - J_m^T * F_h
        #         = tau_m_cmd + delta_tau_s_p + delta_tau_s_f
        #
        # In our sensor-only setup, we don't have F_h directly.
        # F_h is reflected in tau_ext_master (operator presses against arm).
        # The master's own motion reference is q_m (operator's current pose).
        #
        # The MPC computes optimal tau such that the master arm:
        #   a) Tracks a smooth virtual trajectory around q_m
        #   b) Renders force feedback delta_tau_s_f + delta_tau_s_p
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # MASTER IMPEDANCE CONTROL — paper Eq. 9
        #
        # The master renders two feedback signals to the operator:
        #   1. Slave contact force (tau_ext_slave) — operator feels resistance
        #   2. Position tracking error (q_m - q_s) — operator feels slave lag
        #
        # These are combined as additional torques in the master MPC.
        # <<USER: kof scales how strongly slave contact is felt on master.
        #         kom scales how strongly slave position lag is felt.>>
        # ------------------------------------------------------------------
 
        # Aggregate external torque seen by master MPC:
        # tau_ext_for_master = operator force + slave feedback torques
        # <<USER: If you add gain scaling between slave and master torque
        #         spaces (due to different motor ratings), apply here>>
        # tau_ext_master_total = (
        #     tau_ext_master                     # Ch4: operator feel
        #     + delta_tau_s_f                    # Ch3: slave contact rendered
        #     + delta_tau_s_p                    # Ch2: position coupling
        # )

        DEAD_ZONE = np.array([0.5, 0.8, 0.5, 0.3, 0.2, 0.2, 0.2, 0.2])

        tau_ext_slave_filtered = tau_ext_slave.copy()
        # Zero out values below noise threshold
        mask = np.abs(tau_ext_slave_filtered) < DEAD_ZONE
        tau_ext_slave_filtered[mask] = 0.0
        # For values above threshold, subtract the dead zone (so feedback starts from zero)
        tau_ext_slave_filtered = np.sign(tau_ext_slave_filtered) * np.maximum(
            np.abs(tau_ext_slave_filtered) - DEAD_ZONE, 0.0
        )

        tau_feedback_force = -self.p.kof * tau_ext_slave_filtered   # Ch3
        # tau_feedback_force    = -self.p.kof * tau_ext_slave        # Ch3
        tau_feedback_position = -self.p.kom * self.p.Kmd @ (q_m - q_s)  # Ch2

        tau_ext_master_total = (
            tau_ext_master            # Ch4: operator's own applied force
            + tau_feedback_force      # Ch3: slave contact rendered on master
            + tau_feedback_position   # Ch2: slave lag rendered on master
        )
 
        # tau_m_cmd, _, m_ok = self.master_mpc.solve(
        #     q_current  = q_m,
        #     dq_current = dq_m,
        #     q_ref      = q_m,           # master tracks operator (current pos)
        #     M          = M_m,
        #     C          = C_m,
        #     G          = G_m,
        #     tau_ext    = tau_ext_master_total,
        # )
        m_ok = True
        # In current mode the arm has no position controller.
        # tau_m_cmd must continuously provide:
        #   1. Gravity compensation — keeps arm from falling
        #   2. Damping — prevents oscillation when operator releases
        #   3. Force feedback — disturbance from slave contact
        #   4. Position lag feedback — resistance when slave can't follow

        # Compute each term separately for diagnostic clarity
        tau_gravity   = G_m                              # holds arm against gravity
        tau_damping   = -self.p.Dmd @ dq_m              # resists fast motion
        tau_contact   = -self.p.kof * tau_ext_slave_filtered   # slave contact feel
        tau_lag       = -self.p.kom * self.p.Kmd @ (q_m - q_s) # slave lag feel

        tau_m_cmd = tau_gravity + tau_damping + tau_contact + tau_lag
        tau_m_cmd = (
            G_m 
            - self.p.Dmd @ dq_m
            +tau_ext_master_total
        )

        # Clamp in Nm before conversion — use motor-safe limits
        # These are much lower than stall torque to avoid violent motion
        SAFE_TAU_MAX = np.array([2.0, 3.0, 2.0, 0.8, 0.4, 0.4, 0.4, 0.4])
        # <<USER: increase gradually; start conservative>>
        tau_m_cmd = np.clip(tau_m_cmd, -SAFE_TAU_MAX, SAFE_TAU_MAX)

        # Log components for tuning
        print(
            f"G={np.round(tau_gravity[:3],2)} "
            f"D={np.round(tau_damping[:3],2)} "
            f"F={np.round(tau_contact[:3],2)} "
            f"L={np.round(tau_lag[:3],2)} "
            f"→ total={np.round(tau_m_cmd[:3],2)}"
        )
        
        if not m_ok or not np.all(np.isfinite(tau_m_cmd)):
            print("Solver falling back")
            # Fallback: gravity compensation only (safe torque mode)
            # <<USER: Optionally add a damping term here for safety>>
            tau_m_cmd = G_m - self.p.Dmd @ dq_m
        if not m_ok:
            print(f"Master MPC failed — using fallback. G_m={np.round(G_m[:3],3)}")
        else:
            print(f"Master MPC ok. tau_cmd={np.round(tau_m_cmd,3)}")

        # Add temporarily in step():
        print(f"G_m[:5]         = {np.round(G_m[:5], 3)}")
        print(f"Dmd@dq_m[:5]    = {np.round((self.p.Dmd @ dq_m)[:5], 3)}")
        print(f"tau_feedback_f  = {np.round(tau_feedback_force[:5], 3)}")
        print(f"tau_feedback_p  = {np.round(tau_feedback_position[:5], 3)}")
        print(f"tau_ext_master  = {np.round(tau_ext_master[:5], 3)}")
        print(f"tau_ext_slave   = {np.round(tau_ext_slave[:5], 3)}")
                
        slave_gripper_ext = slave_state.tau_ext[6] if len(slave_state.tau_ext) > 6 else 0.0
        gripper_feedback_tau = -self.p.kof * slave_gripper_ext

        # Also add gravity comp for gripper (usually small)
        gripper_gravity = G_m[6] if len(G_m) > 6 else 0.0

        gripper_total_tau =  gripper_gravity + gripper_feedback_tau
        # ------------------------------------------------------------------
        # STEP 4: Slave Admittance Law (paper Eq. 5, second-order)
        #
        # M_sd * ddq_e + D_sd * dq_e + K_sd * q_e
        #       = J_s^T * (F_e - F_m_f - F_m_p)
        #
        # where:
        #   q_e = q_s - q_m   (slave tracking error relative to master)
        #   F_e = tau_ext_slave  (contact force from environment)
        #
        # We solve this ODE one step forward using Euler integration:
        #   ddq_e  = M_sd^{-1} * (tau_net - D_sd*dq_e - K_sd*q_e)
        #   dq_e  += dt * ddq_e
        #   q_e   += dt * dq_e
        #   q_s_ref = q_m + q_e
        # ------------------------------------------------------------------

        # REMOVED
        # q_e = q_s - q_m
        # # Net admittance driving torque (all in joint space)
        # # tau_net = env_contact - master_force_reflected - position_coupling
        # tau_net_admittance = (
        #     tau_ext_slave         # environment reaction
        #     - F_m_f               # master operator force forwarded
        #     - self.p.Ksd @ F_m_p  # position error penalty
        # )
 
        # M_sd_inv = np.linalg.pinv(self.p.Msd)
        # ddq_e = M_sd_inv @ (
        #     tau_net_admittance
        #     - self.p.Dsd @ self._dq_slave_int
        #     - self.p.Ksd @ q_e
        # )
 
        # # Integrate admittance ODE
        # self._dq_slave_int += dt * ddq_e
        # q_e_new = q_e + dt * self._dq_slave_int
 
        # # Admittance reference = master position + compliant offset
        # q_slave_admittance_ref = q_m + q_e_new
 
        # ------------------------------------------------------------------
        # STEP 5: Slave MPC
        #
        # Uses the admittance-derived reference q_slave_admittance_ref.
        # The MPC enforces joint limits and optimises over horizon N.
        #
        # tau_ext for slave MPC: environment torques (contact)
        # <<USER: If you want to weight the slave toward the admittance
        #         reference more aggressively, increase Q_s above>>
        # ------------------------------------------------------------------


        # ------------------------------------------------------------------
        # SLAVE MPC — tracks admittance reference q_slave_ref
        #
        # The MPC does NOT recompute compliance — it simply finds the
        # optimal joint-angle trajectory to reach q_slave_ref within the
        # horizon while respecting joint limits and velocity limits.
        #
        # Output: q_slave_cmd — joint angles sent directly to ReactorX150
        # position controller.
        # ------------------------------------------------------------------

        tau_ext_slave_mpc = tau_ext_slave    # environment contact
 
        _, dq_s_opt, s_ok = self.slave_mpc.solve(
            q_current  = q_s,
            dq_current = dq_s,
            q_ref      = q_slave_ref,
            M          = M_s,
            C          = C_s,
            G          = G_s,
            tau_ext    = tau_ext_slave_mpc,
        )

        # Clamp dq_s_opt regardless of s_ok — never trust raw MPC output
        # <<USER: This limit must match your VIPERX300S_DQ_MAX setting>>
        MAX_DQ_CMD = VIPERX300S_DQ_MAX  # rad/s — matches your conservative DQ_MAX setting
        if np.all(np.isfinite(dq_s_opt)):
            dq_s_opt = np.clip(dq_s_opt, -MAX_DQ_CMD, MAX_DQ_CMD)
        else:
            dq_s_opt = np.zeros(self.n)
            s_ok = False
        
        # Guard: check for NaN/inf before using MPC output
        if s_ok and np.all(np.isfinite(dq_s_opt)):

            # Clamp velocity
            MAX_DQ = 0.4   # rad/s
            dq_s_opt = np.clip(dq_s_opt, -MAX_DQ, MAX_DQ)

            # Integrate velocity to get position
            self._q_slave_int += dt * dq_s_opt

            # Soft pull toward reference — prevents drift without hard clipping
            # This is a leaky integrator: error decays toward zero over time
            # <<USER: alpha=0.05 means 5% correction per cycle toward reference
            #         Increase for tighter tracking, decrease for smoother motion>>
            alpha = 0.05
            self._q_slave_int = (1 - alpha) * self._q_slave_int + alpha * q_slave_ref

            # Hard safety clamp only to prevent physically dangerous positions
            # Use current physical slave position, not reference, as anchor
            self._q_slave_int = np.clip(
                self._q_slave_int,
                q_s - 0.5,   # max 0.5 rad from current physical position
                q_s + 0.5,
            )
            q_slave_cmd = self._q_slave_int

            # Hard clamp: slave cannot move more than 0.5 rad from current position
            # This is the ultimate safety net — prevents any single bad solve
            # from sending the arm to a dangerous position
            # self._q_slave_int += dt * dq_s_opt
            # self._q_slave_int = np.clip(
            #     self._q_slave_int,
            #     q_slave_ref - 0.08,
            #     q_slave_ref + 0.08,
            # )
            #  # Also clip to admittance reference neighbourhood
            # # self._q_slave_int = np.clip(
            # #     self._q_slave_int,
            # #     q_slave_ref - 0.05,
            # #     q_slave_ref + 0.05,
            # # )

            # # alpha = 0.85   # <<USER: tune; higher = tighter reference tracking>>
            # # self._q_slave_int = alpha * q_slave_ref + (1.0 - alpha) * (self._q_slave_int + dt * dq_s_opt)

            # q_slave_cmd = self._q_slave_int
        else:
            # MPC failed — hold slave at its current position, re-sync integrator
            # <<USER: This is the safe fallback. The slave holds still rather than
            #         flying to a random position. Investigate why MPC is failing
            #         by checking the diagnostics topic for solve times.>>
            q_slave_cmd = q_s.copy()
            self._q_slave_int = q_s.copy()   # re-sync so next step starts clean

        # Same guard for master torque:
        if not m_ok or not np.all(np.isfinite(tau_m_cmd)):
            tau_m_cmd = G_m   # gravity compensation only, no C term (C is noisy)
        


        # Simplification 
        # tau_m_cmd = G_m
        return tau_m_cmd, q_slave_cmd
        # return tau_m_cmd, gripper_feedback_tau, q_slave_cmd
 
    # -----------------------------------------------------------------------
    # Jacobian helper (used for Cartesian force ↔ joint torque mapping)
    # -----------------------------------------------------------------------
 
    def jacobian_transpose(self, q: np.ndarray, arm_type: str) -> np.ndarray:
        """
        Returns the Jacobian transpose J^T at configuration q.
 
        <<USER: Implement using your robot's kinematic model.
                Options:
                  - roboticstoolbox: robot.jacob0(q).T
                  - pinocchio: pin.computeJointJacobians(model, data, q); pin.getJointJacobian(...)
                  - ikpy: numerical Jacobian from chain.forward_kinematics_piecewise
 
                Returns (n x 6) matrix (n=5 active DOF, 6 = Cartesian wrench)>>
        """
        # <<USER: Replace identity with real Jacobian>>
        with open(r"/home/sp/Desktop/6th_Semester/RnD/ws/src/controller/controller/data.json") as f:
            d = json.load(f)
            if (arm_type=="viperx300s"):
                return d['vx300s']['J']
            
            return d['rx150']['J']
        