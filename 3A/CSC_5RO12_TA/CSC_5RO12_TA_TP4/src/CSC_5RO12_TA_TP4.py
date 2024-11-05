"""
Extended Kalman Filter SLAM example

author: Atsushi Sakai (@Atsushi_twi)

Modified : Goran Frehse, David Filliat
"""

import math

from dataclasses import dataclass
from math import sin, cos, atan2, pi
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import PercentFormatter

import matplotlib.pyplot as plt
import numpy as np
import os

seed = 123456
np.random.seed(seed)
import os
try:
    os.makedirs("../outputs")
except:
    pass

DT = 0.1  # time tick [s]
simulation_duration = 80  # simulation time [s]
MAX_RANGE = 10.0  # maximum observation range
M_DIST_TH = 9.0  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM state size [x,y]
KNOWN_DATA_ASSOCIATION = 1  # Whether we use the true landmarks id or not

# Simulation parameter
Q_sim = (3 * np.diag([0.1, np.deg2rad(1)])) ** 2    # noise on control input
Py_sim = (1 * np.diag([0.1, np.deg2rad(5)])) ** 2   # noise on measurement

# Kalman filter Parameters
Q = 2 * Q_sim   # Estimated input noise for Kalman Filter
Py = 2 * Py_sim # Estimated measurement noise for Kalman Filter

# Initial estimate of pose covariance


# True Landmark id for known data association
trueLandmarkId = []

# Init displays
# show_animation = True
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(16, 8))
ax3 = plt.subplot(3, 2, 2)
ax4 = plt.subplot(3, 2, 4)
ax5 = plt.subplot(3, 2, 6)






# --- Helper functions

def calc_n_lm(x):
    """
    Computes the number of landmarks in state vector
    """

    n = int((len(x) - STATE_SIZE) / LM_SIZE)
    return n


def calc_landmark_position(x, y):
    """
    Computes absolute landmark position from robot pose and observation
    """

    y_abs = np.zeros((2, 1))

    y_abs[0, 0] = x[0, 0] + y[0] * math.cos(x[2, 0] + y[1])
    y_abs[1, 0] = x[1, 0] + y[0] * math.sin(x[2, 0] + y[1])

    return y_abs


def get_landmark_position_from_state(x, ind):
    """
    Extract landmark position from state vector
    """

    lm = x[STATE_SIZE + LM_SIZE * ind: STATE_SIZE + LM_SIZE * (ind + 1), :]

    return lm


def pi_2_pi(angle):
    """
    Put an angle between -pi / pi
    """

    return (angle + math.pi) % (2 * math.pi) - math.pi


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
        print('Pb with Pxy :\n',Pxy)
        exit()

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [3*a * math.cos(it) for it in t]
    y = [3*b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[bigind, 1], eigvec[bigind, 0])
    rot = np.array([[math.cos(angle), math.sin(angle)],
                    [-math.sin(angle), math.cos(angle)]])
    fx = rot @ (np.array([x, y]))
    px = np.array(fx[0, :] + x_estimation[0, 0]).flatten()
    py = np.array(fx[1, :] + x_estimation[1, 0]).flatten()
    axes.plot(px, py, lineType)


# --- Motion model related functions

def calc_input():
    """
    Generate a control vector to make the robot follow a circular trajectory
    """

    v = 1  # [m/s]
    yaw_rate = 0.1  # [rad/s]
    u = np.array([[v, yaw_rate]]).T
    return u


def motion_model(x, u):
    """
    Compute future robot position from current position and control
    """

    xp = np.array([[x[0,0] + u[0,0]*DT * math.cos(x[2,0])],
                  [x[1,0] + u[0,0]*DT * math.sin(x[2,0])],
                  [x[2,0] + u[1,0]*DT]])
    xp[2] = pi_2_pi(xp[2])

    return xp.reshape((3, 1))


def jacob_motion(x, u):
    """
    Compute the jacobians of motion model wrt x and u
    """

    # Jacobian of f(X,u) wrt X
    A = np.array([[1.0, 0.0, float(-DT * u[0,0] * math.sin(x[2, 0]))],
                  [0.0, 1.0, float(+DT * u[0,0] * math.cos(x[2, 0]))],
                  [0.0, 0.0, 1.0]])

    # Jacobian of f(X,u) wrt u
    B = np.array([[float(DT * math.cos(x[2, 0])), 0.0],
                  [float(DT * math.sin(x[2, 0])), 0.0],
                  [0.0, DT]])

    return A, B


# --- Observation model related functions

def observation(x_true, xd, uTrue, landmarks):
    """
    Generate noisy control and observation and update true position and dead reckoning
    """
    x_true = motion_model(x_true, uTrue)

    # add noise to gps x-y
    y = np.zeros((0, 3))

    for i in range(len(landmarks[:, 0])):

        dx = landmarks[i, 0] - x_true[0, 0]
        dy = landmarks[i, 1] - x_true[1, 0]
        d = math.hypot(dx, dy)
        angle = pi_2_pi(math.atan2(dy, dx) - x_true[2, 0])
        if d <= MAX_RANGE:
            dn = d + np.random.randn() * Py_sim[0, 0] ** 0.5  # add noise
            dn = max(dn,0)
            angle_n = angle + np.random.randn() * Py_sim[1, 1] ** 0.5  # add noise
            yi = np.array([dn, angle_n, i])
            y = np.vstack((y, yi))

    # add noise to input
    u = np.array([[
        uTrue[0, 0] + np.random.randn() * Q_sim[0, 0] ** 0.5,
        uTrue[1, 0] + np.random.randn() * Q_sim[1, 1] ** 0.5]]).T

    xd = motion_model(xd, u)

    return x_true, y, xd, u


def search_correspond_landmark_id(x_estimation, P_estimation, yi):
    """
    Landmark association with Mahalanobis distance
    """

    nLM = calc_n_lm(x_estimation)

    min_dist = []

    for i in range(nLM):
        innovation, S, H = calc_innovation(x_estimation, P_estimation, yi, i)
        min_dist.append(innovation.T @ np.linalg.inv(S) @ innovation)


    min_dist.append(M_DIST_TH)  # new landmark

    min_id = min_dist.index(min(min_dist))

    return min_id


def jacob_h(q, delta, x, i):
    """
    Compute the jacobian of observation model
    """

    sq = math.sqrt(q)
    G = np.array([[-sq * delta[0, 0], - sq * delta[1, 0], 0, sq * delta[0, 0], sq * delta[1, 0]],
                  [delta[1, 0], -delta[0, 0], -q,  -delta[1, 0], delta[0, 0]]])

    G = G / q
    nLM = calc_n_lm(x)
    F1 = np.hstack((np.eye(3), np.zeros((3, 2 * nLM))))
    F2 = np.hstack((np.zeros((2, 3)), np.zeros((2, 2 * i)),
                    np.eye(2), np.zeros((2, 2 * nLM - 2 * (i + 1)))))

    F = np.vstack((F1, F2))

    H = G @ F

    return H


def jacob_augment(x, y):
    """
    Compute the jacobians for extending covariance matrix
    """

    Jr = np.array([[1.0, 0.0, -y[0] * math.sin(x[2,0] + y[1])],
                   [0.0, 1.0, y[0] * math.cos(x[2,0] + y[1])]])

    Jy = np.array([[math.cos(x[2,0] + y[1]), -y[0] * math.sin(x[2,0] + y[1])],
                   [math.sin(x[2,0] + y[1]), +y[0] * math.cos(x[2,0] + y[1])]])

    return Jr, Jy


# --- Kalman filter related functions

def calc_innovation(x_estimation, P_estimation, y, LMid):
    """
    Compute innovation and Kalman gain elements
    """

    # Compute predicted observation from state
    lm = get_landmark_position_from_state(x_estimation, LMid)
    delta = lm - x_estimation[0:2]
    q = (delta.T @ delta)[0, 0]
    y_angle = math.atan2(delta[1, 0], delta[0, 0]) - x_estimation[2, 0]
    yp = np.array([[math.sqrt(q), pi_2_pi(y_angle)]])

    # compute innovation, i.e. diff with real observation
    innovation = (y - yp).T
    innovation[1] = pi_2_pi(innovation[1])

    # compute matrixes for Kalman Gain
    H = jacob_h(q, delta, x_estimation, LMid)
    S = H @ P_estimation @ H.T + Py

    return innovation, S, H


def ekf_slam(x_estimation, P_estimation, u, y):
    """
    Apply one step of EKF predict/correct cycle
    """

    S = STATE_SIZE

    # Predict
    A, B = jacob_motion(x_estimation[0:S], u)

    x_estimation[0:S] = motion_model(x_estimation[0:S], u)

    P_estimation[0:S, 0:S] = A @ P_estimation[0:S, 0:S] @ A.T + B @ Q @ B.T
    P_estimation[0:S,S:] = A @ P_estimation[0:S,S:]
    P_estimation[S:,0:S] = P_estimation[0:S,S:].T

    P_estimation = (P_estimation + P_estimation.T) / 2.0  # ensure symetry

    # Update
    for iy in range(len(y[:, 0])):  # for each observation
        nLM = calc_n_lm(x_estimation)

        if KNOWN_DATA_ASSOCIATION:
            try:
                min_id = trueLandmarkId.index(y[iy, 2])
            except ValueError:
                min_id = nLM
                trueLandmarkId.append(y[iy, 2])
        else:
            min_id = search_correspond_landmark_id(x_estimation, P_estimation, y[iy, 0:2])


        # Extend map if required
        if min_id == nLM:
            print("New LM")

            # Extend state and covariance matrix
            x_estimation = np.vstack((x_estimation, calc_landmark_position(x_estimation, y[iy, :])))

            Jr, Jy = jacob_augment(x_estimation[0:3], y[iy, :])
            bottomPart = np.hstack((Jr @ P_estimation[0:3, 0:3], Jr @ P_estimation[0:3, 3:]))
            rightPart = bottomPart.T
            P_estimation = np.vstack((np.hstack((P_estimation, rightPart)),
                              np.hstack((bottomPart,
                              Jr @ P_estimation[0:3, 0:3] @ Jr.T + Jy @ Py @ Jy.T))))

        else:
            # Perform Kalman update
            innovation, S, H = calc_innovation(x_estimation, P_estimation, y[iy, 0:2], min_id)
            K = (P_estimation @ H.T) @ np.linalg.inv(S)

            x_estimation = x_estimation + (K @ innovation)

            P_estimation = (np.eye(len(x_estimation)) - K @ H) @ P_estimation
            P_estimation = 0.5 * (P_estimation + P_estimation.T)  # Ensure symetry

    x_estimation[2] = pi_2_pi(x_estimation[2])

    return x_estimation, P_estimation




@dataclass
class EKF_SLAM:
    pass



@dataclass
class Utils:
    def compute_motion(x: np.ndarray[float], u: np.ndarray[float], dt: float) -> np.ndarray[float]:
        """

        Compute system motion based on motion equation as np.ndarray.

        Args:
            x (np.ndarray[float]) : system state at instant k-1.
            u (np.ndarray[float]) : control input, or odometry measurement, at instant k.
            dt (float) : simulation time step in seconds.
        """
        assert x.ndim == 2
        assert u.ndim == 2

        x, y, theta = x[0, 0], x[1, 0], x[2, 0]
        vx, vy, omega = u[0, 0], u[1, 0], u[2, 0]

        x_compute = np.array([
            [x + (vx * np.cos(theta) - vy * np.sin(theta)) * dt],
            [y + (vx * np.sin(theta) + vy * np.cos(theta)) * dt],
            [theta + omega * dt],
        ])
        x_compute[2, 0] = Utils.convert_angle(x_compute[2, 0])

        return x_compute


    def convert_angle(angle: float) -> float:
        """
        Return wraped randian angle to the range [-pi, pi].

        Args:
            angle (float): angle in radians.
        """
        if (angle > np.pi):
            angle = angle - 2 * pi
        elif (angle < -np.pi):
            angle = angle + 2 * pi
        return angle


    def compute_ess(weights_particles: np.ndarray[float]) -> float:
        """
        Calculate the Effective Sample Size (ESS).

        Args:
            weights (np.ndarray): particles weights.
        """
        return 1 / np.sum(np.square(weights_particles))


    def compute_rms_error(arr: np.ndarray[float]) -> float:
        """
        Return RMS error of an np.ndarray as a float.

        Args:
            arr (np.ndarray) : array to analyze.
        """
        return np.sqrt(np.mean(arr**2))



class Simulation:
    def __init__(
            self,
            simulation_duration,
            dt_prediction,
            landmarks,
            x_estimation,
            x_odometry,
            x_true,
            P_estimation,
        ) -> None:
        self.simulation_duration = simulation_duration
        self.landmarks = landmarks
        self.dt_prediction = dt_prediction

        self.x_odometry = x_odometry
        self.x_true = x_true

        self.history_x_estimation = x_estimation
        self.history_x_odometry = x_odometry
        self.history_x_true = x_true

        self.history_x_error = np.abs(x_estimation - x_true)
        self.history_x_covariance = np.sqrt(
            np.diag(P_estimation[0:STATE_SIZE, 0:STATE_SIZE]).reshape(3,1)
        )
        self.history_time = [0]


    def plot(self, x_estimation, P_estimation, legend: bool) -> None:
        ax1.cla()

        # Plot true landmark and trajectory
        ax1.plot(self.landmarks[:, 0], self.landmarks[:, 1], "*k")
        ax1.plot(self.history_x_true[0, :], self.history_x_true[1, :], "-k", label="Trajectory")

        # Plot odometry trajectory
        ax1.plot(self.history_x_odometry[0, :], self.history_x_odometry[1, :], "-g", label="Odometry")

        # Plot estimated trajectory, pose and landmarks
        ax1.plot(self.history_x_estimation[0, :], self.history_x_estimation[1, :], "-y", label="EKF")
        ax1.plot(x_estimation[0], x_estimation[1], ".r")
        plot_covariance_ellipse(x_estimation[0: STATE_SIZE],
                                P_estimation[0: STATE_SIZE, 0: STATE_SIZE], ax1, "--r")

        for i in range(calc_n_lm(x_estimation)):
            id = STATE_SIZE + i * 2
            ax1.plot(x_estimation[id], x_estimation[id + 1], "xr")
            plot_covariance_ellipse(x_estimation[id:id + 2],
                                    P_estimation[id:id + 2, id:id + 2], ax1, "--r")




        ax1.grid(True)
        x_max, x_min = +25, -25
        y_max, y_min = +35, -15
        x_ticks = [i for i in range(x_min, x_max+1, 5)]
        y_ticks = [i for i in range(y_min, y_max+1, 5)]
        ax1.axis([x_min, x_max, y_min, y_max])
        ax1.set_title('Cartesian Coordinates')
        ax1.set_ylabel('y [m]')
        ax1.set_xlabel('x [m]')
        ax1.set_xticks(x_ticks)
        ax1.set_yticks(y_ticks)
        ax1.legend(loc='upper left')

        ax2.set_xticks([])


        x_ticks = [i for i in range(0, self.simulation_duration*10+1, 50)]

        # rms_y_err = Utils.compute_rms_error(history_x_error[1, :])
        # rms_theta_err = Utils.compute_rms_error(history_x_error[2, :])

        # rms_y_cov = Utils.compute_rms_error(3*history_x_covariance[1, :])
        # rms_theta_cov = Utils.compute_rms_error(3*history_x_covariance[2, :])

        # label_y_err = f'{rms_y_err:2.4f} error'
        # label_theta_err = f'{rms_theta_err:2.4f} error'

        # label_y_cov = f'{rms_y_cov:2.4f} 3$\sigma$ covariance'
        # label_theta_cov = f'{rms_theta_cov:2.4f} 3$\sigma$ covariance'


        # plot errors curves
        if legend:
            rms_x_err = Utils.compute_rms_error(self.history_x_error[0, :])
            rms_x_cov = Utils.compute_rms_error(3*self.history_x_covariance[0, :])
            label_x_err = f'{rms_x_err:2.4f} error'
            label_x_cov = f'{rms_x_cov:2.4f} 3$\sigma$ covariance'
            ax3.plot(self.history_x_error[0, :],'b', label=label_x_err)
            ax3.plot(+3.0 * self.history_x_covariance[0, :],'r', label=label_x_cov)
        else:
            ax3.plot(self.history_x_error[0, :],'b')
            ax3.plot(+3.0 * self.history_x_covariance[0, :],'r')
        ax3.plot(-3.0 * self.history_x_covariance[0, :],'r')
        ax3.fill_between(
            self.history_time,
            +3.0 * self.history_x_covariance[0, :],
            -3.0 * self.history_x_covariance[0, :],
            color='gray',
            alpha=0.75
        )
        ax3.set_title('Extended Kalman Filter SLAM')
        ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax3.set_ylabel('x [m]')
        ax3.set_xlim(0, simulation_duration)
        ax3.set_xticks(x_ticks)
        ax3.set_xticklabels(['' for _ in x_ticks])
        if legend: ax3.legend(loc='upper right')
        ax3.grid(True)

        ax4.plot(self.history_x_error[1, :],'b')
        ax4.plot(+3.0 * self.history_x_covariance[1, :],'r')
        ax4.plot(-3.0 * self.history_x_covariance[1, :],'r')
        ax4.fill_between(
            self.history_time,
            +3.0 * self.history_x_covariance[1, :],
            -3.0 * self.history_x_covariance[1, :],
            color='gray',
            alpha=0.75
        )
        ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax4.set_ylabel('y [m]')
        ax4.set_xlim(0, simulation_duration)
        ax4.set_xticks(x_ticks)
        ax4.set_xticklabels(['' for _ in x_ticks])
        # ax4.legend(loc='upper right')
        ax4.grid(True)

        ax5.plot(self.history_x_error[2, :],'b')
        ax5.plot(+3.0 * self.history_x_covariance[2, :],'r')
        ax5.plot(-3.0 * self.history_x_covariance[2, :],'r')
        ax5.fill_between(
            self.history_time,
            +3.0 * self.history_x_covariance[2, :],
            -3.0 * self.history_x_covariance[2, :],
            color='gray',
            alpha=0.75
        )
        ax5.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax5.set_ylabel(r"$\theta$ [rad]")
        ax5.set_xlabel('time [s]')
        ax5.set_xlim(0, simulation_duration)
        ax5.set_xticks(x_ticks)
        # ax5.legend(loc='upper right')
        ax5.grid(True)

        plt.tight_layout()
        plt.pause(0.001)

        # ax3.set_label(label_x_err)



    def update_history(
            self, k: int, x_estimation: np.ndarray[float], P_estimation: np.ndarray[float]
        ) -> None:
        """
        Update simulation history.

        Args:
            k (int) : simulation instant.
            x_estimation (np.ndarray[float]) : system state estimation at instant k.
            P_estimation (np.ndarray[float]) : estimation covariance at instant k.
        """
        self.history_x_true = np.hstack((self.history_x_true, self.x_true))
        self.history_x_odometry = np.hstack((self.history_x_odometry, self.x_odometry))
        self.history_x_estimation = np.hstack((self.history_x_estimation, x_estimation[0:STATE_SIZE]))

        error = x_estimation[0:STATE_SIZE] - self.x_true
        error[2, 0] = pi_2_pi(error[2, 0])

        self.history_x_error = np.hstack((self.history_x_error, error))
        self.history_x_covariance = np.hstack((
            self.history_x_covariance, np.sqrt(
                np.diag(P_estimation[0:STATE_SIZE, 0:STATE_SIZE]).reshape(3,1)
            )
        ))
        self.history_time.append(k*10)









def execution(
        landmarks: np.ndarray[float],
        dt_prediction: int = 1,
        P_constant: int = 1,
        save: bool = False,
        show: bool = True,
    ) -> None:
    """
    Execute an Particles Filter simulation for SLAM.

    Args:
        save (bool) : save result? Default value is False.
        show (bool) : show result? Default value is True.
    """
    # Simulation initial conditions
    P_true = np.diag([0.01, 0.01, 0.0001])
    P_estimation = P_constant * P_true

    x_true = np.zeros((STATE_SIZE, 1))
    x_odometry = np.zeros((STATE_SIZE, 1))
    x_estimation = np.zeros((STATE_SIZE, 1))


    simulation = Simulation(
        simulation_duration,
        dt_prediction,
        landmarks,
        x_estimation,
        x_odometry,
        x_true,
        P_estimation,
    )

    # counter for plotting
    count = 0
    time = 0.0

    # todo
    #   remove global scoope
    #   create classes
    #   document EKF functions

    # TODO create plot init function
    #   include supname
    #   include axis from global scope on local scope

    while  time <= simulation.simulation_duration:
        # simulation.simulate_world(k)
        count = count + 1
        time += DT

        # Simulate motion and generate u and y
        # TODO update observation mode
        uTrue = calc_input()
        x_true, y, x_odometry, u = observation(x_true, x_odometry, uTrue, landmarks)
        x_estimation, P_estimation = ekf_slam(x_estimation, P_estimation, u, y)

        simulation.x_true = x_true
        simulation.x_odometry = x_odometry

        simulation.update_history(time, x_estimation, P_estimation)

        if show and count%10==0:
            simulation.plot(x_estimation, P_estimation, legend=False)

    # todo modify legend variable name
    #   use for to set plot.
    simulation.plot(x_estimation, P_estimation, legend=True)

    file_name = f'EKF_SLAM_{dt_prediction}_'
    file_name += f'{landmarks.shape[0]}_'
    file_name += f'{P_constant}.png'
    file_path = os.path.join(os.path.abspath(os.getcwd()), '../outputs', file_name)

    plt.suptitle(file_name)
    if save: plt.savefig(file_path, dpi=300)
    if show: plt.show()



def main():
    landmarks = np.array([
        [+00.0, +05.0],
        [+11.0, +01.0],
        [+03.0, +15.0],
        [-05.0, +20.0]
    ])

    # remove 
    execution(landmarks=landmarks, save=True, show=True)



if __name__ == '__main__':
    main()
