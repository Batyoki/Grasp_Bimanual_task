"""
arm_interface.py
----------------
Abstraction layer for the two robot types used in the bilateral
teleoperation system:

  Master: ViperX 300s   (6-DOF physical, 5-DOF with wrist-roll locked)
  Slave:  ReactorX 150  (5-DOF physical)

Both arms share the same 5-DOF active joint space after constraining
the wrist-roll joint (joint index 5, 0-indexed).

Motor Sensor Availability
--------------------------
The ONLY sensors available are the Dynamixel motor's internal current
(and hence torque) readings, plus joint position from encoders.
There are NO external F/T sensors.

Torque Estimation
-----------------
  tau_measured = kt * I_measured  (per joint)
  where kt is the motor torque constant.

  The net EXTERNAL torque (due to contact) is estimated as:
  tau_ext = tau_measured - (M * ddq + C + G)

  Since ddq must be numerically differentiated from encoder data,
  apply a low-pass filter to avoid amplifying noise.

Mass / Coriolis / Gravity
--------------------------
Dynamics are estimated using pinochio

"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import pinocchio as pin
import json


# ---------------------------------------------------------------------------
# Joint limits (radians) — adjust to match your hardware config
# ---------------------------------------------------------------------------

# <<USER: Verify these limits against your ViperX300s URDF / spec sheet>>
VIPERX300S_Q_MIN = np.concatenate([
    np.array([-180, -101, -101, -107, -180, -180]) * (np.pi / 180),
    np.array([0.042, -0.042])
])
VIPERX300S_Q_MAX = np.concatenate([
    np.array([180, 101, 92, 130, 180, 180]) * (np.pi / 180),
    np.array([0.116, -0.116])
])
# NOT Verified
# VIPERX300S_DQ_MAX = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0])   # rad/s or prismatic speed for fingers


# VERY HIGH TORQUE VALUES
# VIPERX300S_TAU_MAX = np.array([8.0, 10.0, 8.0, 5.0, 3.0, 3.0, 5.0, 5.0])    # N·m or finger effort estimate
VIPERX300S_TAU_MAX = np.array([3.0, 5.0, 3.0, 1.0, 0.5, 0.5, 1.0, 1.0])    # N·m or finger effort estimate


# <<USER: Verify these limits against your ReactorX150 URDF / spec sheet>>
REACTORX150_Q_MIN = np.concatenate([
    np.array([-180, -108, -108, -100, -180, -180]) * (np.pi / 180),
    np.array([0.030, -0.030])
])
REACTORX150_Q_MAX = np.concatenate([
    np.array([180, 113, 93, 123, 180, 180]) * (np.pi / 180),
    np.array([0.074, -0.074])
])

# NOT VERIFIED

# REACTORX150_DQ_MAX = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0]) #rad/s
# Very HIGH
# REACTORX150_TAU_MAX = np.array([5.0, 6.0, 5.0, 3.5, 2.5, 2.5, 5.0, 5.0]) #N.m
REACTORX150_TAU_MAX = np.array([3.0, 4.0, 3.0, 1.5, 0.5, 0.5, 1.0, 1.0]) #N.m

# VIPERX300S_DQ_MAX = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])  # rad/s
# REACTORX150_DQ_MAX = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])


VIPERX300S_DQ_MAX = np.array([0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.1, 0.1])  # rad/s
REACTORX150_DQ_MAX = np.array([0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.1, 0.1])

# ---------------------------------------------------------------------------
# Motor torque constants (Nm/A) for each joint's Dynamixel motor
# <<USER: Fill in actual kt values from your Dynamixel motor spec sheets>>
# For XM430 series: kt ~ 2.0–2.5 Nm/A (depends on winding)
# For XL430 / XC430 series: kt ~ 1.5–1.8 Nm/A
# ---------------------------------------------------------------------------

VIPERX300S_KT   = np.array([2.409, 2.409, 2.409, 2.409, 2.409, 1.783 ,1.783 ,1.783])  # <<USER: verify>>
REACTORX150_KT  = np.array([1.783, 1.783, 1.783, 1.071, 1.071, 1.071, 1.071, 1.071])  # <<USER: verify>>

MAX_CURRENT_MA = np.array([26.9, 26.9, 26.9, 18.0, 18.0])

# Wrist-roll joint index (0-based) that is LOCKED / excluded
WRIST_ROLL_IDX = 3  # joint6 in full 6-DOF chain; not used in 5-DOF model
ACTIVE_JOINTS  = [0, 1, 2, 3, 4]  # waist, shoulder, elbow, wrist roll, pitch

RX200_GRIPPER_CLOSED = -0.9    # <<USER: replace with your observed closed value>>
RX200_GRIPPER_OPEN   = 1.79    # <<USER: replace with your observed open value>>
VX300S_GRIPPER_CLOSED = 0.037 # closed = fingers together
VX300S_GRIPPER_OPEN   = -0.037 # open = fingers apart

default_viper_forearm_roll_angle = 180.7 * np.pi/180


@dataclass
class ArmState:
    """Snapshot of a single arm's sensed state at one time step."""
    q:          np.ndarray           # Joint positions  (5,)
    dq:         np.ndarray           # Joint velocities (5,) — numerical diff
    tau_motor:  np.ndarray           # Motor torques from current sensing (5,)
    tau_ext:    np.ndarray           # Estimated external contact torques (5,)
    timestamp:  float = 0.0          # ROS time in seconds


class ArmDynamics:
    """
    Holds and evaluates joint-space dynamics for one arm.

    The user must provide callables (or pre-computed matrices) for:
      M(q)  — inertia matrix
      C(q, dq) — Coriolis/centrifuge vector
      G(q)  — gravity vector

    These can be generated from:
      - roboticstoolbox  (DHRobot.inertia / .coriolis / .gravload)
      - ikpy + custom differentiator
      - pinocchio library (recommended for real-time use)
      - a symbolic model exported from Matlab/Simulink

    <<USER: Replace lambda stubs below with your actual dynamic model>>
    """

    def __init__(self, arm_type: str):
        assert arm_type in ("viperx300s", "reactorx150")
        self.arm_type = "vx300s" if arm_type == "viperx300s" else "rx150"
        self.n = 8  # active DOF

        json_data=None
        with open(r'/home/sp/Desktop/6th_Semester/RnD/ws/src/controller/controller/data.json','r') as f:
            json_data = json.load(f)
        
        self.M_fn  = lambda q:       np.asarray(json_data[self.arm_type]['M'], dtype=float)
        self.C_fn  = lambda q, dq:   self.get_C(q,dq)
        self.G_fn  = lambda q:       self.get_G(q)

        Murdf_path = "/home/sp/Desktop/6th_Semester/RnD/ws/src/controller/controller/rx200.urdf"
        self.Mmodel, self.Mcollision_model, self.Mvisual_model = pin.buildModelsFromUrdf(Murdf_path)        
        self.Mdata = self.Mmodel.createData()

        surdf_path = '/home/sp/Desktop/6th_Semester/RnD/ws/src/controller/controller/vx300s.urdf'
        self.Smodel, self.Scollision_model, self.Svisual_model = pin.buildModelsFromUrdf(surdf_path)        
        self.Sdata = self.Smodel.createData()

        # In ArmDynamics.__init__, add after building both models:
        # print(f"[ArmDynamics] vx300s: nq={self.Smodel.nq}, nv={self.Smodel.nv}")
        # print(f"[ArmDynamics] rx200:  nq={self.Mmodel.nq}, nv={self.Mmodel.nv}")
        # Expected: vx300s nq=6, rx200 nq=5
        # If different, your URDF has extra fixed joints — adjust q[:nq] accordingly

        # print('Slave')
        # print(self.Smodel.nq)
        # print('Master')
        # print(self.Mmodel.nq)

        
        # def printsss(model):
        #     idx = 0
        #     for i in range(1, model.njoints):
        #         joint = model.joints[i]
        #         name = model.names[i]
        #         nq = joint.nq  # DOFs for this joint

        #         print(f"{name}: q[{idx}:{idx+nq}]")

        #         idx += nq
        # printsss(self.Mmodel)
        # print('\n\n\n')
        # printsss(self.Smodel)


    def get_M(self, q: np.ndarray) -> np.ndarray:
        M = self.M_fn(q)
        if M.ndim == 2 and M.shape[0] != self.n:
            M = M[: self.n, : self.n]
        return M

    def get_C(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """
        Returns the Coriolis/centrifuge torque VECTOR: C(q,dq) @ dq
        Shape: (n,) matching the arm's active DOF
        """
        if self.arm_type == "vx300s":
            # Extend to full pinocchio DOF by inserting forearm_roll
            q_full  = np.insert(q,  3, 0.0)   # shape (8,) for pinocchio
            dq_full = np.insert(dq[:7], 3, 0.0)   # <<USER: use 0 vel for locked joint>>

            q_full  = np.asarray(q_full,  dtype=np.float64)
            dq_full = np.asarray(dq_full, dtype=np.float64)

            # C_mat is (nv x nv) — multiply by dq to get torque vector
            C_mat = pin.computeCoriolisMatrix(self.Smodel, self.Sdata, q_full, dq_full)
            C_vec_full = C_mat @ dq_full           # shape (nv,) — actual torque vector

            # Remove the forearm_roll entry (index 3) to get back to 7 active joints
            C_vec = np.delete(C_vec_full, 3)       # shape (7,) or (nv-1,)

        else:  # rx150/rx200
            # print(q,dq)
            q_full  = np.insert(q,  5, 0.0)
            dq_full = np.insert(dq[:7], 5, 0.0)

            q_full  = np.asarray(q_full,  dtype=np.float64)
            dq_full = np.asarray(dq_full, dtype=np.float64)

            # print(q_full,dq_full)

            C_mat = pin.computeCoriolisMatrix(self.Mmodel, self.Mdata, q_full, dq_full)
            C_vec_full = C_mat @ dq_full
            C_vec = np.delete(C_vec_full, 5)

        # Pad or truncate to self.n (8 DOF including gripper/fingers)
        result = np.zeros(self.n)
        result[:len(C_vec)] = C_vec[:self.n]
        return result

    def get_G(self, q: np.ndarray) -> np.ndarray:
        if(self.arm_type=="vx300s"):
            # q here is 8-DOF (arm+gripper). Pinocchio vx300s model has 7 arm joints.
            # Take first 7 (arm only), insert forearm_roll at index 3
            q_arm   = q                             # 7 arm joints from sensor
            q_full  = np.insert(q_arm, 3, 0.0)            # insert forearm_roll=0
            q_full  = np.asarray(q_full, dtype=np.float64)

            G_full = pin.computeGeneralizedGravity(self.Smodel, self.Sdata, q_full)
            G_arm  = np.delete(G_full, 3)  
        else:
            # rx150: insert a fixed joint value to match pinocchio model's expected DOF
            q_arm  = q
            q_full = np.insert(q_arm, 5, 0.0)             # rx200: fixed joint at index 5
            q_full = np.asarray(q_full, dtype=np.float64)

            G_full = pin.computeGeneralizedGravity(self.Mmodel, self.Mdata, q_full)
            G_arm  = np.delete(G_full, 5)

        result = np.zeros(self.n)
        result[:len(G_arm)] = G_arm[:self.n]
        return result



class ArmInterface:
    """
    Wraps a physical arm's state estimation and torque sensing.

    Responsibilities
    ----------------
    1. Read joint positions from Dynamixel encoders (via ROS topic).
    2. Read motor current → estimate joint torque.
    3. Numerically differentiate joint positions → velocity.
    4. Estimate external torques by subtracting model torques.
    5. Apply low-pass filters to all noisy signals.
    """

    def __init__(
        self,
        arm_type: str,                # "viperx300s" or "reactorx150"
        dynamics: ArmDynamics,
        dt: float,
        filter_alpha: float = 0.3,   # 1st-order IIR smoothing coeff
    ):
        assert arm_type in ("viperx300s", "reactorx150")
        self.arm_type = "vx300s" if arm_type == "viperx300s" else "rx150"
        self.dynamics = dynamics
        self.dt = dt
        self.alpha = filter_alpha
        # self.n = 5

        # dof changed to 8
        self.n = 8

        # Motor torque constants
        self.kt = (VIPERX300S_KT if arm_type == "viperx300s"
                   else REACTORX150_KT)

        # Internal state buffers
        self._q_prev:   Optional[np.ndarray] = np.zeros(self.n)
        self._dq_filt:  np.ndarray = np.zeros(self.n)
        self._tau_filt: np.ndarray = np.zeros(self.n)

        self._initialised = False

    # ------------------------------------------------------------------
    # Raw sensor ingestion
    # ------------------------------------------------------------------

    def update(
        self,
        q_raw: np.ndarray,          # Raw encoder positions (8,)  [rad]
        current_raw: np.ndarray,    # Raw motor currents   (8,)  [A]
    ) -> ArmState:
        """
        Process one sensor snapshot and return an ArmState.

        Call this at each control loop iteration.
        """
        q = q_raw.copy()
        current = current_raw.copy()
        
        # Handle case where joint state has fewer values than expected
        # Pad with zeros if necessary
        if len(q) < self.n:
            q = np.concatenate([q, np.zeros(self.n - len(q))])
        elif len(q) > self.n:
            q = q[:self.n]  # Truncate if too many
            
        if len(current) < self.n:
            current = np.concatenate([current, np.zeros(self.n - len(current))])
        elif len(current) > self.n:
            current = current[:self.n]

        # --- Velocity: numerical differentiation + IIR low-pass -------
        if self._q_prev is None or not self._initialised:
            self._initialised=True
            self._q_prev = q.copy()
            dq_raw = np.zeros(self.n)
            # dq_raw = q/self.dt
        else:
            assert len(self._q_prev) == self.n and len(q) == self.n, \
                f"Joint state size mismatch: expected {self.n}, got {len(q)}"
            dq_raw = (q - self._q_prev) / self.dt

        # IIR: dq_filt[k] = alpha*dq_raw + (1-alpha)*dq_filt[k-1] #low pass filter
        self._dq_filt = self.alpha * dq_raw + (1 - self.alpha) * self._dq_filt
        self._q_prev  = q.copy()

        # --- Torque from current: tau = kt * I -------------------------
        # <<USER: Verify sign conventions for your Dynamixel setup.
        #         Some joints may need sign flip depending on mounting.>>
        tau_raw = self.kt * current
        self._tau_filt = (self.alpha * tau_raw
                          + (1 - self.alpha) * self._tau_filt)

        # --- External torque estimation --------------------------------
        # tau_measured = M*ddq + C + G + tau_ext
        # => tau_ext = tau_measured - M*ddq - C - G
        #
        # We skip ddq (double diff is too noisy without filtering) and
        # approximate tau_ext as the residual between measured torque
        # and the gravity + Coriolis model only.
        # For better accuracy: add a state observer / Kalman filter here.
        # <<USER: If you implement ddq estimation, uncomment M*ddq term>>

        M = np.array(self.dynamics.get_M(q))
        C = np.array(self.dynamics.get_C(q, self._dq_filt))
        G = np.array(self.dynamics.get_G(q))

        tau_model = C + G  # + M @ ddq_estimated  #(<<USER: add if available>>)
        tau_ext   = self._tau_filt - tau_model


        # # UNTIL JOINT ANGLE MIMICING WORKS, TAU_EXT IS SET TO 0
        # tau_ext = np.zeros(self.n)

        return ArmState(
            q=q,
            dq=self._dq_filt.copy(),
            tau_motor=self._tau_filt.copy(),
            tau_ext=tau_ext,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def q_limits(self):
        if self.arm_type == "viperx300s":
            return VIPERX300S_Q_MIN, VIPERX300S_Q_MAX
        return REACTORX150_Q_MIN, REACTORX150_Q_MAX

    @property
    def tau_limits(self):
        if self.arm_type == "viperx300s":
            return -VIPERX300S_TAU_MAX, VIPERX300S_TAU_MAX
        return -REACTORX150_TAU_MAX, REACTORX150_TAU_MAX

    @property
    def dq_limits(self):
        if self.arm_type == "viperx300s":
            return -VIPERX300S_DQ_MAX, VIPERX300S_DQ_MAX
        return -REACTORX150_DQ_MAX, REACTORX150_DQ_MAX