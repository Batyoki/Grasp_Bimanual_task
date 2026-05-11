import pinocchio as pin
import numpy as np

# Load modified URDF (already 5 DOF)
urdf_path = "/home/sp/Desktop/6th_Semester/RnD/ws/src/controller/controller/dump/rx200.urdf"
model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()

# -------------------------------
# State
# -------------------------------
q = pin.randomConfiguration(model)   # joint positions
v = np.zeros(model.nv)               # velocities
a = np.zeros(model.nv)               # accelerations

# -------------------------------
# Dynamics
# -------------------------------

# Inertia matrix M(q)
pin.crba(model, data, q)
M = (data.M + data.M.T) / 2

# Coriolis matrix C(q, v)
pin.computeCoriolisMatrix(model, data, q, v)
C = data.C
c = C @ v

# Gravity vector g(q)
g = pin.computeGeneralizedGravity(model, data, q)

# Inverse dynamics tau
tau = pin.rnea(model, data, q, v, a)

frame_name = "rx200/gripper_link"

frame_id = model.getFrameId(frame_name)
J = pin.computeFrameJacobian(
        model,
        data,
        q,
        frame_id,
        pin.ReferenceFrame.WORLD
    ).tolist()
print(f'J = {J}')

# -------------------------------
# Debug / sanity check
# -------------------------------
print("DOF:", model.nv)
print("M shape:", M.shape)
print("C shape:", C.shape)
print("g shape:", g.shape)
print("tau shape:", tau.shape)

# Verify dynamics consistency
tau_check = M @ a + C @ v + g
print("Consistency error:", np.linalg.norm(tau - tau_check))
print(M.tolist())
print(C)
print(g)


np.savez(
    "dynamics_data_rx200.txt",
    M=M,
    C=C,
    g=g,
    tau=tau,
    q=q,
    v=v,
    a=a
)