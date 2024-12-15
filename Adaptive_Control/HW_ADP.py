import numpy as np
import os
from simulator import Simulator
from pathlib import Path
from typing import Dict
import pinocchio as pin
import matplotlib.pyplot as plt
from scipy.linalg import logm

current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
model_1 = pin.buildModelFromMJCF(xml_path)
data_1 = model_1.createData()

times_1 = []
positions_1 = []
velocities_1 = []
control_1 = []
error_1 = []

pi_last = np.zeros(60,dtype=float)
    
def plot_results(times: np.ndarray, positions: np.ndarray, velocities: np.ndarray, torque: np.ndarray, error: np.ndarray):
    """Plot and save simulation results."""
    # Joint positions plot
    plt.figure(figsize=(10, 6))
    for i in range(positions.shape[1]):
        plt.plot(times, positions[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Positions [rad]')
    plt.title('Joint Positions over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/HW_ADP_pos.png')
    plt.close()
    
    # Joint velocities plot
    plt.figure(figsize=(10, 6))
    for i in range(velocities.shape[1]):
        plt.plot(times, velocities[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Velocities [rad/s]')
    plt.title('Joint Velocities over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/HW_ADP_vel.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    for i in range(velocities.shape[1]):
        plt.plot(times, torque[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Torque [Nm]')
    plt.title('Joint Torque over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/HW_ADP_tor.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    for i in range(positions.shape[1]):
        plt.plot(times, error[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Positions Error [rad]')
    plt.title('Joint Positions Error over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/HW_ADP_err.png')
    plt.close()

def skew_to_vector(m):
    return np.array([m[2,1], m[0,2], m[1,0]])

def R_err(Rd, R):
    error_matrix = Rd @ R.T
    error_log = logm(error_matrix)
    return skew_to_vector(error_log)
    # return pin.log3(Rd @ R.T)

def traj(t):
    R = 0.2
    X = np.array([0.15+R*np.sin(2*np.pi*t/2), -0.7+R*np.cos(2*np.pi*t/2), 0.5+R*np.cos(2*np.pi*t/2), 0, 0., -np.pi/2])
    dX = np.array([2*np.pi*R*np.cos(2*np.pi*t/2), -2*np.pi*R*np.sin(2*np.pi*t/2), -R*2*np.pi*np.sin(2*np.pi*t/2), 0, 0, 0])
    ddX = np.array([-((2*np.pi)**2)*R*np.sin(2*np.pi*t/2), -((2*np.pi)**2)*R*np.cos(2*np.pi*t/2), -R*((2*np.pi)**2)*np.cos(2*np.pi*t/2), 0, 0, 0])

    return X, dX, ddX

def calcInertia(q, dq):
    pin.forwardKinematics(model_1, data_1, q, dq)
    pin.updateFramePlacements(model_1, data_1)

    pi = model_1.inertias[0].toDynamicParameters()
    for i in range(1,5):
        pi = np.hstack((pi, model_1.inertias[i].toDynamicParameters()))
    return pi

def joint_controller(q: np.ndarray, dq: np.ndarray, t: float) -> np.ndarray:
    """Example task space controller."""

    global pi_last

    times_1.append(t)
    positions_1.append(q)
    velocities_1.append(dq)

    q_d = np.array([-1.4, -1.3, 1., 0., 0., 0.])
    dq_d = np.array([0., 0., 0., 0., 0., 0.])
    ddq_d = np.array([0., 0., 0., 0., 0., 0.])

    q_d, dq_d, ddq_d = traj(t)

    print(q_d - q)
    error_1.append(q_d - q)

    lambda_1 = np.array([6., 20., 35., 10, 12., 10.])

    s = (dq_d - dq) + lambda_1*(q_d - q)

    # Control gains tuned for UR5e
    kp = np.array([100, 100, 100, 100, 100, 100])
    kd = np.array([50, 50, 50, 80, 50, 50])

    regressor = pin.computeJointTorqueRegressor(model_1, data_1, q, dq, ddq_d+lambda_1*(dq_d - dq)+ kd*s)

    pi_last += 0.00000005*(regressor.T @ s)*0.002

    pi_1 = np.hstack((calcInertia(q, dq), pi_last[-10:]))

    print(pi_1[-10:])

    tau = regressor @ pi_1 

    control_1.append(tau)

    return tau

def main():
    # Create logging directories
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    
    print("\nRunning task space controller...")
    sim = Simulator(
        xml_path="robots/universal_robots_ur5e/scene.xml",
        enable_task_space=False,
        show_viewer=True,
        record_video=True,
        video_path="logs/videos/HW_ADP.mp4",
        fps=30,
        width=1920,
        height=1080
    )

    ee_name = "end_effector"

    sim.modify_body_properties(ee_name, mass=3)
    # Print modified properties
    props = sim.get_body_properties(ee_name)
    print(f"\nModified end-effector properties:")
    print(f"Mass: {props['mass']:.3f} kg")
    print(f"Inertia:\n{props['inertia']}")

    sim.set_controller(joint_controller)

    sim.run(time_limit=5.0)

if __name__ == "__main__":
    # print(current_dir)
    main() 

    times_1 = np.array(times_1)
    positions_1 = np.array(positions_1)
    velocities_1 = np.array(velocities_1)
    control_1 = np.array(control_1)
    error_1 = np.array(error_1)

    plot_results(times_1, positions_1, velocities_1, control_1, error_1)

    