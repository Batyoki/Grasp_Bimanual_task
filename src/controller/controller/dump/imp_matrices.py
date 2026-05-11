import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import numpy as np
import json


file_path = r'~/Workspaces/interbotix_ws/src/interbotix_ros_manipulators/interbotix_ros_xsarms/interbotix_xsarm_descriptions'
meshes=f"{file_path}/meshes/"
urdf_path = f"{file_path}/urdf/"

vx300s = {
    'meshes' : f"{meshes}vx300s_meshes/",
    'urdf' : f"./vx300s.urdf"
    }

rx150 = {
    'meshes' : f"{meshes}rx200_meshes/",
    'urdf' : f"rx200.urdf"
    }

vx300s_arm = RobotWrapper.BuildFromURDF(vx300s['urdf'],vx300s['meshes'])
rx150_arm = RobotWrapper.BuildFromURDF(rx150['urdf'],rx150['meshes'])

vx300s_model = vx300s_arm.model
vx300s_data = vx300s_arm.data

rx150_model = rx150_arm.model
rx150_data = rx150_arm.data

l=np.array([1,2,3])
l=l.tolist()
print(type(l))

json_file = f"./data.json"

def compute_matrices(model, data, q, qdot,frame_name):
    M = pin.crba(model, data, q).tolist()
    print(M)

    C = pin.computeCoriolisMatrix(model, data, q, qdot).tolist()
    print(C)

    g = pin.computeGeneralizedGravity(model, data, q).tolist()
    print(g)

    

    
    frame_id = model.getFrameId(frame_name)

    pin.forwardKinematics(model,data,q)
    pin.updateFramePlacements(model,data)

    J = pin.computeFrameJacobian(
        model,
        data,
        q,
        frame_id,
        pin.ReferenceFrame.WORLD
    ).tolist()
    print(J)

    return [M,C,g,J]

q = np.zeros(vx300s_model.nq)
qdot = np.zeros(vx300s_model.nv)
frame_name = "vx300s/gripper_link"
M,C,g,J = compute_matrices(vx300s_model,vx300s_data,q,qdot,frame_name)
vx300s['M']=M
vx300s['C']=C
vx300s['g']=g
vx300s['J']=J

print("RX200")
q = np.zeros(rx150_model.nq)
qdot = np.zeros(rx150_model.nv)
frame_name = "rx200/gripper_link"
M,C,g,J = compute_matrices(rx150_model,rx150_data,q,qdot,frame_name)
rx150['M']=M
rx150['C']=C
rx150['g']=g
rx150['J']=J


with open(json_file,"w") as f:
    json.dump({
        'vx300s':vx300s,
        "rx200":rx150
    },f)

