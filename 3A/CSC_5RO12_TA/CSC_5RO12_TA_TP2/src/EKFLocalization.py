"""
TP Kalman filter for mobile robots localization

authors: Goran Frehse, David Filliat, Nicolas Merlinge
"""

from math import sin, cos, atan2, pi, sqrt
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import numpy as np
seed = 123456
np.random.seed(seed)

import os
try:
    os.makedirs("outputs")
except:
    pass

# ---- Simulator class (world, control and sensors) ----

class Simulation:
    def __init__(self, time_final, dt_prediction, x_true, Q_true, x_odometry, landmarks, R_true, dt_measure):
        self.time_final = time_final
        self.dt_prediction = dt_prediction
        self.n_steps = int(np.round(time_final/dt_prediction))
        self.Q_true = Q_true
        self.x_true = x_true
        self.x_odometry = x_odometry
        self.landmarks = landmarks
        self.R_true = R_true
        self.dt_measure = dt_measure

    # return true control at step k
    def get_robot_control(self, k):
        # generate  sin trajectory
        u = np.array([[0, 0.025,  0.1*np.pi / 180 * sin(3*np.pi * k / self.n_steps)]]).T
        return u


    # simulate new true robot position
    def simulate_world(self, k):
        dt_prediction = self.dt_prediction
        u = self.get_robot_control(k)
        self.x_true = tcomp(self.x_true, u, dt_prediction)
        self.x_true[2, 0] = convert_angle(self.x_true[2, 0])


    # computes and returns noisy odometry
    def get_odometry(self, k):
        # Ensuring random repetability for given k
        np.random.seed(seed*2 + k)

        # Model
        dt_prediction = self.dt_prediction
        u = self.get_robot_control(k)
        xnow = tcomp(self.x_odometry, u, dt_prediction)
        uNoise = np.sqrt(self.Q_true) @ np.random.randn(3)
        uNoise = np.array([uNoise]).T
        xnow = tcomp(xnow, uNoise, dt_prediction)
        self.x_odometry = xnow
        u = u + dt_prediction*uNoise
        return xnow, u


    # generate a noisy observation of a random feature
    def get_observation(self, k):
        # Ensuring random repetability for given k
        np.random.seed(seed*3 + k)

        # Model
        if k*self.dt_prediction % self.dt_measure == 0:
            notValidCondition = False # False: measurement valid / True: measurement not valid
            if notValidCondition:
                z = None
                landmark_index = None
            else:
                landmark_index = np.random.randint(0, self.landmarks.shape[1] - 1)
                zNoise = np.sqrt(self.R_true) @ np.random.randn(2)
                zNoise = np.array([zNoise]).T
                z = observation_model_prediction(self.x_true, landmark_index, self.landmarks) + zNoise
                z[1, 0] = convert_angle(z[1, 0])
        else:
            z = None
            landmark_index = None
        return [z, landmark_index]


# ---- Kalman Filter: model functions ----

# evolution model (f)
def motion_model_prediction(x: np.ndarray[float], u: np.ndarray[float], dt: float) -> np.ndarray[float]:
    """
    Returns prediction of system state x at instant k knowing instant k-1.

    Args:
        x (np.ndarray[float]) : system state at instant k-1
        u (np.ndarray[float]) : 
    """
    # x: estimated state (x, y, heading)
    # u: control input or odometry measurement in body frame (Vx, Vy, angular rate)

    # print(f'{type(x)}')
    x_k, y_k, theta_k = x[0, 0], x[1, 0], x[2, 0]
    vx_k, vy_k, w_k = u[0, 0], u[1, 0], u[2, 0]

    x_pred = np.zeros_like(x)
    x_pred[0, 0] = x_k + (vx_k * np.cos(theta_k) - vy_k * np.sin(theta_k)) * dt
    x_pred[1, 0] = y_k + (vx_k * np.sin(theta_k) + vy_k * np.cos(theta_k)) * dt
    x_pred[2, 0] = theta_k + w_k * dt
    x_pred[2, 0] = convert_angle(x_pred[2, 0]) # convert angle to [-pi,+pi]

    return x_pred


# observation model (h)
def observation_model_prediction(x: np.ndarray[float], i: int, landmarks: np.ndarray[float]) -> np.ndarray[float]:
    # xVeh: vecule state
    # landmark_index: observed guide index
    # landmarks: landmarks of all guides

    x_k, y_k, theta_k = x[0, 0], x[1, 0], x[2, 0]
    x_p, y_p = landmarks[0, i], landmarks[1, i]

    y = np.array([
        [np.sqrt((x_p - x_k)**2 + (y_p - y_k)**2)],
        [np.arctan2((y_p - y_k), (x_p - x_k)) - theta_k],
    ])
    y[1, 0] = convert_angle(y[1, 0])

    return y


# ---- Kalman Filter: Jacobian functions ----

# h(x) Jacobian wrt x
def observation_jacobian(x: np.ndarray[float], i: int, landmarks: np.ndarray[float]) -> np.ndarray[float]:
    """
    Return Jacobian Observation matrix H(x) as np.ndarray[float].

    Args:
        x (np.ndarray[float]) : system state prediction.
        i (int) : landmark index.
        landmarks (np.ndarray[float]) : landmarks coordinates.
    """
    x_k, y_k = x[0, 0], x[1, 0]
    x_p, y_p = landmarks[0, i], landmarks[1, i]

    delta_x = x_p - x_k
    delta_y = y_p - y_k
    distance = np.sqrt(delta_x**2 + delta_y**2)

    H = np.array([
        [-delta_x/distance, -delta_y/distance, 0],
        [+delta_y/distance**2, -delta_x/distance**2, -1]
    ])

    return H


# f(x,u) Jacobian wrt x
# jacobian of motion model w.r.t state
def F(x: np.ndarray[float], u: np.ndarray[float], dt: float) -> np.ndarray[float]:
    # x: estimated state (x, y, heading)
    # u: control input (Vx, Vy, angular rate)
    # dt_prediction: time step

    _, _, theta_k = x[0, 0], x[1, 0], x[2, 0]
    vx_k, vy_k, _ = u[0, 0], u[1, 0], u[2, 0]

    F = np.array([
        [1, 0, (-vx_k * np.sin(theta_k) - vy_k * np.cos(theta_k)) * dt],
        [0, 1, (+vx_k * np.cos(theta_k) - vy_k * np.sin(theta_k)) * dt],
        [0, 0, 1]
    ])

    return F


# f(x,u) Jacobian wrt w (noise on the control input u)
# jacobian of motion model w.r.t control
def G(x: np.ndarray[float], u: np.ndarray[float], dt: float) -> np.ndarray[float]:
    # x: estimated state (x, y, heading) in ground frame
    # u: control input (Vx, Vy, angular rate) in robot frame
    # dt_prediction: time step for prediction

    theta_k = x[2, 0]

    G = np.array([
        [np.cos(theta_k) * dt, -np.sin(theta_k) * dt, 0],
        [np.sin(theta_k) * dt, +np.cos(theta_k) * dt, 0],
        [0, 0, dt]
    ])

    return G


# ---- Utils functions ----
# Display error ellipses
def plot_covariance_ellipse(x_estimation, P_estimation, axes, lineType):
    """
    Plot one covariance ellipse from covariance matrix
    """

    Pxy = P_estimation[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    if eigval[smallind] < 0:
        print('Pb with Pxy :\n', Pxy)
        exit()

    t = np.arange(0, 2 * pi + 0.1, 0.1)
    a = sqrt(eigval[bigind])
    b = sqrt(eigval[smallind])
    x = [3 * a * cos(it) for it in t]
    y = [3 * b * sin(it) for it in t]
    angle = atan2(eigvec[bigind, 1], eigvec[bigind, 0])
    rot = np.array([[cos(angle), sin(angle)],
                    [-sin(angle), cos(angle)]])
    fx = rot @ (np.array([x, y]))
    px = np.array(fx[0, :] + x_estimation[0, 0]).flatten()
    py = np.array(fx[1, :] + x_estimation[1, 0]).flatten()
    axes.plot(px, py, lineType)


def convert_angle(angle: float) -> float:
    """
    Return converted randian angle to the range [-pi, pi].

    Args:
        angle (float): angle in radians.
    """
    if (angle > np.pi):
        angle = angle - 2 * pi
    elif (angle < -np.pi):
        angle = angle + 2 * pi
    return angle


# composes two transformations
def tcomp(tab, tbc, dt):
    assert tab.ndim == 2 # eg: robot state [x, y, heading]
    assert tbc.ndim == 2 # eg: robot control [Vx, Vy, angle rate]
    #dt : time-step (s)

    angle = tab[2, 0] + dt * tbc[2, 0] # angular integration by Euler

    angle = convert_angle(angle)
    s = sin(tab[2, 0])
    c = cos(tab[2, 0])
    position = tab[0:2] + dt * np.array([[c, -s], [s, c]]) @ tbc[0:2] # position integration by Euler
    out = np.vstack((position, angle))

    return out


# =============================================================================
# Main Program
# =============================================================================

# Init displays
show_animation = True
save_iterations = False

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(16, 8))
ax3 = plt.subplot(3, 2, 2)
ax4 = plt.subplot(3, 2, 4)
ax5 = plt.subplot(3, 2, 6)

# ---- General variables ----

# Simulation time
time_final = 6000       # final time (s)
dt_prediction = 1     # Time between two dynamical predictions (s)
dt_measure = 1     # Time between two measurement updates (s)

# Location of landmarks
n_landmarks = 30
landmarks = 140*(np.random.rand(2, n_landmarks) - 1/2)

# True covariance of errors used for simulating robot movements
Q_true = np.diag([0.01, 0.01, 1*pi/180]) ** 2
R_true = np.diag([3.0, 3*pi/180]) ** 2

# Modeled errors used in the Kalman filter process
Q_constant = 1
R_constant = 1

Q_estimation = Q_constant * np.eye(3, 3) @ Q_true
R_estimation = R_constant * np.eye(2, 2) @ R_true

# initial conditions
x_true = np.array([[1, -40, -pi/2]]).T
x_odometry = x_true
x_estimation = x_true
P_constant = 1
P_estimation = P_constant * np.diag([1, 1, (1*pi/180)**2])

# Init history matrixes
history_x_estimation = x_estimation
history_x_true = x_true
history_x_odometry = x_odometry
history_x_error = np.abs(x_estimation-x_true)  # pose error
history_x_variance = np.sqrt(np.diag(P_estimation).reshape(3, 1))  # state std dev
history_time = [0]

# Simulation environment
simulation = Simulation(time_final, dt_prediction, x_true, Q_true, x_odometry, landmarks, R_true, dt_measure)

# Temporal loop
for k in range(1, simulation.n_steps):

    # Simulate robot motion
    simulation.simulate_world(k)

    # Odometry measurements
    x_odometry, u_tilde = simulation.get_odometry(k)

    # Kalman Prediction
    x_prediction = motion_model_prediction(x_estimation, u_tilde, simulation.dt_prediction)

    F_k = F(x_prediction, u_tilde, simulation.dt_prediction) # TODO pourquoi predction et non estimation ou odometry
    G_k = G(x_prediction, u_tilde, simulation.dt_prediction)
    P_prediction = F_k @ P_estimation @ F_k.T + G_k @ Q_estimation @ G_k.T

    # Random landmark observation
    [y, landmark_index] = simulation.get_observation(k)

    # Kalman Correction
    if y is not None:
        # Observation available
        y_prediction = observation_model_prediction(x_prediction, landmark_index, simulation.landmarks)

        innovation = y - y_prediction
        innovation[1, 0] = convert_angle(innovation[1, 0])

        H = observation_jacobian(x_prediction, landmark_index, simulation.landmarks)
        S = R_estimation + H @ P_prediction @ H.T
        K = P_prediction @ H.T @ np.linalg.inv(S)


        # Kalman Update
        x_estimation = x_prediction + K @ innovation
        x_estimation[2, 0] = convert_angle(x_estimation[2, 0])

        P_estimation = (np.eye(K.shape[0]) - K @ H) @ P_prediction
        P_estimation = 0.5 * (P_estimation + P_estimation.T)  # extract symetry matrix

    else:
        # Observation unavailable
        x_estimation = x_prediction
        P_estimation = P_prediction

    # Store data
    history_x_true = np.hstack((history_x_true, simulation.x_true))
    history_x_odometry = np.hstack((history_x_odometry, simulation.x_odometry))
    history_x_estimation = np.hstack((history_x_estimation, x_estimation))

    error = x_estimation - simulation.x_true
    error[2, 0] = convert_angle(error[2, 0])

    history_x_error = np.hstack((history_x_error, error))
    history_x_variance = np.hstack((history_x_variance, np.sqrt(np.diag(P_estimation).reshape(3, 1))))
    history_time.append(k*simulation.dt_prediction)


    # plot every 15 updates
    if show_animation and k*simulation.dt_prediction % 200 == 0:
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

        ax1.cla()

        times = np.stack(history_time)

        # Plot true landmark and trajectory
        ax1.plot(landmarks[0, :], landmarks[1, :], "*k")
        ax1.plot(history_x_true[0, :], history_x_true[1, :], "-k", label="Trajectory")

        # Plot odometry trajectory
        ax1.plot(history_x_odometry[0, :], history_x_odometry[1, :], "-g", label="Odometry")

        # Plot estimated trajectory an pose covariance
        ax1.plot(history_x_estimation[0, :], history_x_estimation[1, :], "-y", label="Extended Kalman Filter")
        ax1.plot(x_estimation[0], x_estimation[1], ".r")
        plot_covariance_ellipse(x_estimation, P_estimation, ax1, "--r")

        ax1.grid(True)
        ax1.axis([-100, 100, -100, 100])
        ax1.set_title('Cartesian Coordinates')
        ax1.set_ylabel('y [m]')
        ax1.set_xlabel('x [m]')
        ax1.legend(loc='upper left')

        # add common error markers
        ax2.set_xticks([])

        # plot errors curves
        ax3.plot(times, history_x_error[0, :], 'b')
        ax3.plot(times, +3.0 * history_x_variance[0, :], 'r')
        ax3.plot(times, -3.0 * history_x_variance[0, :], 'r')
        ax3.set_title('Error [blue] and 3$\sigma$ Covariances [red]')
        ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax3.set_ylabel('x [m]')
        ax3.set_xlim(0, 6000)
        ax3.set_xticks([0, 1000, 2000, 3000, 4000, 5000, 6000])
        ax3.set_xticklabels(['', '', '', '', '', '', ''])
        ax3.grid(True)

        ax4.plot(times, history_x_error[1, :], 'b')
        ax4.plot(times, +3.0 * history_x_variance[1, :], 'r')
        ax4.plot(times, -3.0 * history_x_variance[1, :], 'r')
        ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax4.set_ylabel('y [m]')
        ax4.set_xlim(0, 6000)
        ax4.set_xticks([0, 1000, 2000, 3000, 4000, 5000, 6000])
        ax4.set_xticklabels(['', '', '', '', '', '', ''])
        ax4.grid(True)

        ax5.plot(times, history_x_error[2, :], 'b')
        ax5.plot(times, +3.0 * history_x_variance[2, :], 'r')
        ax5.plot(times, -3.0 * history_x_variance[2, :], 'r')
        ax5.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax5.set_ylabel(r"$\theta$ [rad]")
        ax5.set_xlabel('time [s]')
        ax5.set_xlim(0, 6000)
        ax5.set_xticks([0, 1000, 2000, 3000, 4000, 5000, 6000])
        ax5.grid(True)

        plt.tight_layout()

        if save_iterations: plt.savefig(r'outputs/EKF_' + str(k) + '.png')
#        plt.pause(0.001)

plt.savefig(os.path.join(os.path.abspath(os.getcwd()), 'outputs', f'EKF_{Q_constant}_{R_constant}_{P_constant}.png'), dpi=300)
plt.show()
