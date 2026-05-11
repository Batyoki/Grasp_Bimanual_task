"""
mpc_optimizer.py
----------------
MPC (Model Predictive Control) optimizer for bilateral teleoperation.

The MPC problem is formulated as a finite-horizon quadratic program:

    min  sum_{k=0}^{N-1} [ (q_k - q_ref)^T Q (q_k - q_ref)
                           + (dq_k)^T R (dq_k)
                           + (tau_k)^T S (tau_k) ]
         + (q_N - q_ref)^T P (q_N - q_ref)

    s.t. q_{k+1} = q_k + dt * dq_k          (Euler kinematics)
         dq_{k+1} = dq_k + dt * M^{-1}(tau_k - C - G - tau_ext)
         q_min  <= q_k  <= q_max
         dq_min <= dq_k <= dq_max
         tau_min<= tau_k<= tau_max

The optimizer is used for BOTH master and slave arms, but with
different reference signals and objective weights:
  - Master: tracks operator intent, renders feedback torques
  - Slave:  tracks master joint angles, complies with environment forces

"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Optional
import pinocchio as pin


class MPCOptimizer:
    """
    Finite-horizon MPC optimizer using scipy's SQP solver.

    For real-time deployment consider replacing scipy with a dedicated
    QP solver such as OSQP or quadprog for lower latency.
    """

    def __init__(
        self,
        n_joints: int,
        horizon: int,
        dt: float,
        Q: np.ndarray,   # State tracking weight  [n x n]
        R: np.ndarray,   # Control effort weight  [n x n]
        S: np.ndarray,   # Torque magnitude weight[n x n]
        P: np.ndarray,   # Terminal state weight  [n x n]
        q_min: np.ndarray,
        q_max: np.ndarray,
        dq_min: np.ndarray,
        dq_max: np.ndarray,
        tau_min: np.ndarray,
        tau_max: np.ndarray,
    ):
        self.n = n_joints
        self.N = horizon
        self.dt = dt
        self.Q = Q
        self.R = R
        self.S = S
        self.P = P
        self.q_min = q_min
        self.q_max = q_max
        self.dq_min = dq_min
        self.dq_max = dq_max
        self.tau_min = tau_min
        self.tau_max = tau_max

        # SANITY CHECK ASSERTIONS
        for name, arr in [("Q", Q), ("R", R), ("S", S), ("P", P)]:
            assert arr.shape == (n_joints, n_joints), \
                f"Weight matrix {name} shape {arr.shape} != ({n_joints},{n_joints})"
        for name, arr in [("q_min", q_min), ("q_max", q_max),
                        ("dq_min", dq_min), ("dq_max", dq_max),
                        ("tau_min", tau_min), ("tau_max", tau_max)]:
            assert arr.shape == (n_joints,), \
                f"Limit array {name} shape {arr.shape} != ({n_joints},)"

        # Decision variable size: [dq_increments(N*n), tau_commands(N*n)]
        self._n_dec = 2 * self.N * self.n

        # Warm-start storage
        self._last_solution: Optional[np.ndarray] = None


        

    def _build_bounds(self):
        """Build scipy bounds for the decision variables."""
        lb_dq  = np.tile(self.dq_min,  self.N)
        ub_dq  = np.tile(self.dq_max,  self.N)
        lb_tau = np.tile(self.tau_min, self.N)
        ub_tau = np.tile(self.tau_max, self.N)
        lb = np.concatenate([lb_dq, lb_tau])
        ub = np.concatenate([ub_dq, ub_tau])
        return list(zip(lb, ub))

    def _rollout(
        self,
        x0: np.ndarray,          # [q0; dq0]
        dec_vars: np.ndarray,    # flattened [dq_inc; tau]
        M_inv: np.ndarray,       # M^{-1}
        C: np.ndarray,           # Coriolis vector
        G: np.ndarray,           # Gravity vector
        tau_ext: np.ndarray,     # External / feedback torque
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Roll out the joint-space dynamics over the MPC horizon.

        Returns
        -------
        qs  : (N+1, n) joint position trajectory
        dqs : (N+1, n) joint velocity trajectory
        """
        dq_seq  = dec_vars[: self.N * self.n].reshape(self.N, self.n)
        tau_seq = dec_vars[self.N * self.n :].reshape(self.N, self.n)

        qs  = np.zeros((self.N + 1, self.n))
        dqs = np.zeros((self.N + 1, self.n))
        qs[0]  = x0[: self.n]
        dqs[0] = x0[self.n :]

        for k in range(self.N):
            # Joint-space Newton-Euler dynamics (linearised, Euler step):
            #   M ddq + C + G = tau + tau_ext
            #   ddq = M^{-1}(tau_k - C - G + tau_ext)
            # -------------------------------------------------------
            # NOTE: C and G here are evaluated at the current step's
            # q/dq.  For higher fidelity, recompute them at each step
            # using your robot's dynamic model (see arm_interface.py).
            # -------------------------------------------------------
            ddq = M_inv @ (tau_seq[k] - C - G + tau_ext)
            # dqs[k + 1] = dqs[k] + self.dt * ddq
            # qs[k + 1]  = qs[k]  + self.dt * dqs[k] + 0.5 * self.dt**2 * ddq

            if not np.all(np.isfinite(ddq)) or np.any(np.abs(ddq) > 1e4):
            # Signal infeasibility by returning current state repeated
                qs[k+1:]  = qs[k]
                dqs[k+1:] = 0.0
                break

            dqs[k + 1] = dqs[k] + self.dt * ddq
            qs[k + 1]  = qs[k]  + self.dt * dqs[k] + 0.5 * self.dt**2 * ddq

        return qs, dqs

    def _objective(
        self,
        dec_vars: np.ndarray,
        x0: np.ndarray,
        q_ref: np.ndarray,
        M_inv: np.ndarray,
        C: np.ndarray,
        G: np.ndarray,
        tau_ext: np.ndarray,
    ) -> float:
        dq_seq  = dec_vars[: self.N * self.n].reshape(self.N, self.n)
        tau_seq = dec_vars[self.N * self.n :].reshape(self.N, self.n)

        qs, _ = self._rollout(x0, dec_vars, M_inv, C, G, tau_ext)

        cost = 0.0
        for k in range(self.N):
            eq = qs[k] - q_ref
            if not np.all(np.isfinite(eq)):
                return 1e10
            cost += eq @ self.Q @ eq
            cost += dq_seq[k] @ self.R @ dq_seq[k]
            cost += tau_seq[k] @ self.S @ tau_seq[k]

        # Terminal cost
        eq_N = qs[self.N] - q_ref
        cost += eq_N @ self.P @ eq_N
        return cost

    def solve(
        self,
        q_current: np.ndarray,   # Current joint positions (n,)
        dq_current: np.ndarray,  # Current joint velocities (n,)
        q_ref: np.ndarray,       # Reference joint positions (n,)
        M: np.ndarray,           # Mass matrix  (n x n)
        C: np.ndarray,           # Coriolis vector (n,)
        G: np.ndarray,           # Gravity vector  (n,)
        tau_ext: np.ndarray,     # External torque (feedback) (n,)
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Run one MPC solve step.

        Returns
        -------
        tau_cmd : (n,) optimal torque command for the first step
        dq_cmd  : (n,) optimal velocity increment for the first step
        success : bool
        """
        M_inv = np.linalg.pinv(M)   # Use pinv for numerical safety
        x0 = np.concatenate([q_current, dq_current])

        # Warm-start: shift previous solution by one step; zero-pad tail
        if self._last_solution is not None:
            n = self.n
            N = self.N
            dq_prev = self._last_solution[: N * n].reshape(N, n)
            tau_prev = self._last_solution[N * n :].reshape(N, n)
            dq_warm  = np.vstack([dq_prev[1:],  np.zeros((1, n))])
            tau_warm = np.vstack([tau_prev[1:], np.zeros((1, n))])
            x0_opt = np.concatenate([dq_warm.ravel(), tau_warm.ravel()])
        else:
            x0_opt = np.zeros(self._n_dec)

        bounds = self._build_bounds()

        result = minimize(
            self._objective,
            x0_opt,
            args=(x0, q_ref, M_inv, C, G, tau_ext),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 50, "ftol": 1e-6},
        )

        success = result.success or result.fun < 1e-3  # accept near-converged
        if success:
            self._last_solution = result.x

        tau_cmd = result.x[self.N * self.n :][: self.n]
        dq_cmd  = result.x[: self.n]

        # Sanitise before returning — NaN in result.x means optimizer diverged
        if not np.all(np.isfinite(tau_cmd)) or not np.all(np.isfinite(dq_cmd)):
            return np.zeros(self.n), np.zeros(self.n), False
        
        return tau_cmd, dq_cmd, success