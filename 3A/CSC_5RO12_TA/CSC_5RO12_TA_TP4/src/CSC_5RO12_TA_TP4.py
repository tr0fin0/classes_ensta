"""
[CSC_5RO12_TA_TP4] Navegation pour la Robotique, Extended Kalman Filter SLAM

author: Atsushi Sakai (@Atsushi_twi)
modified : Goran Frehse, David Filliat
modified : Guilherme Trofino
"""
from dataclasses import dataclass
from matplotlib.ticker import FormatStrFormatter

import math
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



S_S = 3  # state size: [x, y, omega]
L_S = 2  # state size landmarks: [x, y]



@dataclass
class EKF_SLAM:
    """"Extended Kalman Filter for SLAM methods."""

    def F(x: np.ndarray[float], u: np.ndarray[float], dt: float) -> np.ndarray[float]:
        """
        Return motion model state jacobian matrix F(x) as np.ndarray[float].

        Note: jacobian with respect of system state.

        Args:
            x (np.ndarray[float]) : system state at instant k-1.
            u (np.ndarray[float]) : control input, or odometry measurement, at instant k.
            dt (float) : simulation time step in seconds.
        """
        _, _, theta = x[0, 0], x[1, 0], x[2, 0]
        v, _ = u[0, 0], u[1, 0]

        F = np.array([
            [1.0, 0.0, -v * math.sin(theta) * dt],
            [0.0, 1.0, +v * math.cos(theta) * dt],
            [0.0, 0.0, 1.0]
        ])

        return F


    def G(x: np.ndarray[float], u: np.ndarray[float], dt: float) -> np.ndarray[float]:
        """
        Return motion model control jacobian matrix G(x) as np.ndarray[float].

        Note: jacobian with respect of noised odometry.

        Args:
            x (np.ndarray[float]) : system state at instant k-1.
            u (np.ndarray[float]) : control input, or odometry measurement, at instant k.
            dt (float) : simulation time step in seconds.
        """
        _, _, theta = x[0, 0], x[1, 0], x[2, 0]

        G = np.array([
            [math.cos(theta) * dt, 0.0],
            [math.sin(theta) * dt, 0.0],
            [0.0, dt]
        ])

        return G


    def search_landmark_id(
            yi: np.ndarray[float],
            x_estimation: np.ndarray[float],
            P_estimation: np.ndarray[float],
            R_estimation: np.ndarray[float],
            mahalanobis_distance: float = 9.0
        ):
        """
        Return landmark id with Mahalanobis distance.

        Args:
            yi (np.ndarray[float]) : 
            x_estimation (np.ndarray[float]) : system state estimation at instant k.
            P_estimation (np.ndarray[float]) : system state estimation covariance at instant k.
            R_estimation (np.ndarray[float]) : measure noise covariance matrix at instant k.
            mahalanobis_distance (float) : mahalanobis threshold distance. Default value is 9.0.
        """
        dist_min = []

        for i in range(Utils.count_landmarks(x_estimation)):
            innovation, S, _ = EKF_SLAM.calc_innovation(
                i,
                yi,
                x_estimation,
                P_estimation,
                R_estimation,
            )
            dist_min.append(innovation.T @ np.linalg.inv(S) @ innovation)

        dist_min.append(mahalanobis_distance)

        return dist_min.index(min(dist_min))


    def extended_jacobians(
            x_estimation: np.ndarray[float], y_estimation: np.ndarray[float]
        ) -> list[np.ndarray[float], np.ndarray[float]]:
        """
        Return exteded covariances jacobians matrices.

        Note: Jr is the state extended jacobian and Jy is the command extended jacobian.

        Args:
            x_estimation (np.ndarray[float]) : system state estimation at instant k.
            y_estimation (np.ndarray): observed landmark estimation at instant k.
        """
        _, _, theta = x_estimation[0, 0], x_estimation[1, 0], x_estimation[2, 0]
        v, w = y_estimation[0], y_estimation[1]


        Jr = np.array([
            [1.0, 0.0, -v * math.sin(theta + w)],
            [0.0, 1.0, +v * math.cos(theta + w)]
        ])

        Jy = np.array([
            [math.cos(theta + w), -v * math.sin(theta + w)],
            [math.sin(theta + w), +v * math.cos(theta + w)]
        ])

        return Jr, Jy


    def H(
            distance: float,
            relative_position: np.ndarray[float],
            x: np.ndarray[float],
            landmark_index: int
        ) -> np.ndarray[float]:
        """
        Return observation model jacobian matrix H(x) for a specific landmark as np.ndarray[float].

        Args:
            distance (float): squared Euclidean distance between the robot and the landmark.
            relative_position (np.ndarray[float]): relative position vector [delta_x, delta_y]^T.
            x (np.ndarray[float]) : system state estimation at instant k.
            landmark_index (int): landmark index in the state vector.
        """
        delta_x, delta_y = relative_position[0, 0], relative_position[1, 0]

        sq = math.sqrt(distance)
        G = np.array([
            [-sq * delta_x, -sq * delta_y, 0, sq * delta_x, sq * delta_y],
            [delta_y, -delta_x, -distance,  -delta_y, delta_x]
        ])
        G = G / distance

        landmark_count = Utils.count_landmarks(x)

        F1 = np.hstack((
            np.eye(3),
            np.zeros((3, 2 * landmark_count))
        ))
        F2 = np.hstack((
            np.zeros((2, 3)),
            np.zeros((2, 2 * landmark_index)),
            np.eye(2),
            np.zeros((2, 2 * landmark_count - 2 * (landmark_index + 1)))
        ))

        F = np.vstack((F1, F2))

        H = G @ F

        return H


    def calc_innovation(
            landmark_index,
            y,
            x_estimation,
            P_estimation,
            R_estimation,
        ):
        """
        Compute innovation vector and related matrices for Kalman gain in EKF-SLAM.

        Args:
            landmark_index (int): Index of the landmark in the state vector.
            y (np.ndarray): observed landmark at instant k.
            x_estimation (np.ndarray[float]) : system state estimation at instant k.
            P_estimation (np.ndarray[float]) : system state estimation covariance at instant k.
            R_estimation (np.ndarray[float]) : measure noise covariance matrix at instant k.
        """
        # Compute predicted observation from state
        landmark_position = Utils.get_landmark_position_state(x_estimation, landmark_index)
        relative_position = landmark_position - x_estimation[0:2]

        distance = (relative_position.T @ relative_position)[0, 0]
        y_angle = math.atan2(relative_position[1, 0], relative_position[0, 0]) - x_estimation[2, 0]

        y_prediction = np.array([[math.sqrt(distance), Utils.convert_angle(y_angle)]])

        # compute innovation, i.e. diff with real observation
        innovation = (y - y_prediction).T
        innovation[1] = Utils.convert_angle(innovation[1])

        # compute matrixes for Kalman Gain
        H = EKF_SLAM.H(distance, relative_position, x_estimation, landmark_index)
        S = H @ P_estimation @ H.T + R_estimation

        return innovation, S, H


    def compute_iteration(
            y: np.ndarray[float],
            u: np.ndarray[float],
            x_estimation: np.ndarray[float],
            P_estimation: np.ndarray[float],
            Q_estimation: np.ndarray[float],
            R_estimation: np.ndarray[float],
            dt: float,
            landmarks_known: bool,
            undelayed_method: bool,
            landmarks_true_id: list,
        ):
        """
        Execute Extended Kalman Filter for SLAM prediction and correction.

        Args:

            u (np.ndarray[float]) : control input, or odometry measurement, at instant k-1.
            x_estimation (np.ndarray[float]) : system state estimation at instant k-1.
            P_estimation (np.ndarray[float]) : system state estimation covariance at instant k-1.
            Q_estimation (np.ndarray[float]) : process noise covariance matrix at instant k-1.
            R_estimation (np.ndarray[float]) : measure noise covariance matrix at instant k-1.
            dt (float) : simulation time step in seconds.

        """
        # Kalman States Prediction
        x_estimation[0:S_S] = Utils.compute_motion(x_estimation[0:S_S], u, dt)

        F = EKF_SLAM.F(x_estimation[0:S_S], u, dt)
        G = EKF_SLAM.G(x_estimation[0:S_S], u, dt)

        P_estimation[0:S_S, 0:S_S] = F @ P_estimation[0:S_S, 0:S_S] @ F.T + G @ Q_estimation @ G.T

        P_estimation[0:S_S, S_S:] = F @ P_estimation[0:S_S, S_S:]
        P_estimation[S_S:, 0:S_S] = P_estimation[0:S_S, S_S:].T

        P_estimation = 0.5 * (P_estimation + P_estimation.T)    # symetry matrix


        # Kalman Correction
        for iy in range(len(y[:, 0])):
            landmark_count = Utils.count_landmarks(x_estimation)

            if landmarks_known:
                try:
                    min_id = landmarks_true_id.index(y[iy, 2])
                except ValueError:
                    min_id = landmark_count
                    landmarks_true_id.append(y[iy, 2])
            else:
                min_id = EKF_SLAM.search_landmark_id(
                    y[iy, 0:2], x_estimation, P_estimation, R_estimation
                )


            # Extend map if required
            if min_id == number_landmarks:
                print("New LM")

                # Extend state and covariance matrix
                x_estimation = np.vstack((x_estimation, Utils.get_landmark_position_absolute(x_estimation, y[iy, :])))

                Jr, Jy = EKF_SLAM.jacob_augment(x_estimation[0:3], y[iy, :])
                bottomPart = np.hstack((Jr @ P_estimation[0:3, 0:3], Jr @ P_estimation[0:3, 3:]))
                rightPart = bottomPart.T
                P_estimation = np.vstack((np.hstack((P_estimation, rightPart)),
                                np.hstack((bottomPart,
                                Jr @ P_estimation[0:3, 0:3] @ Jr.T + Jy @ R_estimation @ Jy.T))))

            else:
                # Perform Kalman update
                innovation, S, H = EKF_SLAM.calc_innovation(x_estimation, P_estimation, y[iy, 0:2], min_id, R_estimation)
                K = (P_estimation @ H.T) @ np.linalg.inv(S)

                x_estimation = x_estimation + (K @ innovation)

                P_estimation = (np.eye(len(x_estimation)) - K @ H) @ P_estimation
                P_estimation = 0.5 * (P_estimation + P_estimation.T)    # symetry matrix

        x_estimation[2] = Utils.convert_angle(x_estimation[2])

        return x_estimation, P_estimation



@dataclass
class Utils:
    def compute_motion(x_estimation: np.ndarray[float], u_tilde: np.ndarray[float], dt: float) -> np.ndarray[float]:
        """
        Compute system motion based on motion equation as np.ndarray.

        Args:
            x_estimation (np.ndarray[float]) : system state at instant k.
            u_tilde (np.ndarray[float]) : control input, or odometry measurement, at instant k.
            dt (float) : simulation time step in seconds.
        """
        assert x_estimation.ndim == 2
        assert u_tilde.ndim == 2

        x, y, theta = x_estimation[0, 0], x_estimation[1, 0], x_estimation[2, 0]
        v, omega = u_tilde[0, 0], u_tilde[1, 0]

        x_motion = np.array([
            [x + v * dt * np.cos(theta)],
            [y + v * dt * np.sin(theta)],
            [theta + omega * dt],
        ])
        x_motion[2, 0] = Utils.convert_angle(x_motion[2, 0])

        return x_motion


    def convert_angle(angle: float) -> float:
        """
        Return wraped randian angle to the range [-pi, pi].

        Args:
            angle (float): angle in radians.
        """
        return (angle + math.pi) % (2 * math.pi) - math.pi


    def compute_rms_error(arr: np.ndarray[float]) -> float:
        """
        Return RMS error of an np.ndarray as a float.

        Args:
            arr (np.ndarray) : array to analyze.
        """
        return np.sqrt(np.mean(arr**2))


    def count_landmarks(x_estimation: np.ndarray[float]) -> int:
        """
        Return quantity of landmarks in the a state vector x.

        Args: 
            x_estimation (np.ndarray[float]) : system state at instant k.
        """
        return int((len(x_estimation) - S_S) / L_S)


    def get_landmark_position_absolute(
            x_estimation: np.ndarray[float],
            u_tilde: np.ndarray[float]
        ) -> np.ndarray[float]:
        """
        Return landmark absolute position from robot pose and get_observation.

        Args:
            x_estimation (np.ndarray[float]) : system state at instant k.
            u_tilde (np.ndarray[float]) : control input, or odometry measurement, at instant k.
        """
        x, y, theta = x_estimation[0, 0], x_estimation[1, 0], x_estimation[2, 0]
        v, omega = u_tilde[0], u_tilde[1]

        landmark_position = np.zeros((2, 1))

        landmark_position[0, 0] = x + v * math.cos(theta + omega)
        landmark_position[1, 0] = y + v * math.sin(theta + omega)

        return landmark_position


    def get_landmark_position_state(x: np.ndarray[float], index: int) -> np.ndarray[float]:
        """
        Return landmark state position from robot pose.

        Args:
            x (np.ndarray[float]) : system state at instant k.
            index (int) : landmark index on system state.
        Extract landmark position from state vector
        """
        return x[
            S_S + L_S * index : S_S + L_S * (index + 1), :
        ]


    def is_initial_position(
            x_initial: np.ndarray[float],
            x_estimation: np.ndarray[float],
            dist_threshold: float = 0.1
        ) -> bool:
        """
        Return if estimated position is initial position as boolean based on a distance threshold.

        Args:
            x_initial (np.ndarray[float]) : system state estimation at instant 0.
            x_estimation (np.ndarray[float]) : system state estimation at instant k.
        """
        x_0, y_0 = x_initial[0, 0], x_initial[1, 0]
        x_k, y_k = x_estimation[0, 0], x_estimation[1, 0]

        return np.sqrt((x_0 - x_k)**2 + (y_0 - y_k)**2) < dist_threshold


    def plot_covariance_ellipse(
            x_estimation: np.ndarray[float],
            P_estimation: np.ndarray[float],
            line: str,
            axes,
            landmarks_data: tuple
        ) -> None:
        """
        Plot motion estimation convariance matrix as an ellipse.

        Args:
            x_estimation (np.ndarray[float]) : system state estimation at instant k.
            P_estimation (np.ndarray[float]) : system state estimation covariance at instant k-1.
            line (str) : plot line sytle.
            axes () : matplotlib.plot axis to include plot.
        """
        P_xy = P_estimation[0:2, 0:2]
        eigen_values, eigen_vectors = np.linalg.eig(P_xy)

        if eigen_values[0] >= eigen_values[1]:
            index_big, index_small = 0, 1
        else:
            index_big, index_small = 1, 0

        if eigen_values[index_small] < 0:
            print('Pb with P_xy :\n',P_xy)
            exit()

        a = math.sqrt(eigen_values[index_big])
        b = math.sqrt(eigen_values[index_small])
        area = math.pi * a * b

        circle = np.arange(0, 2 * math.pi + 0.1, 0.1)
        x = [3 * a * math.cos(angle) for angle in circle]
        y = [3 * b * math.sin(angle) for angle in circle]

        angle = math.atan2(eigen_vectors[index_big, 1], eigen_vectors[index_big, 0])
        rotation = np.array([
            [+math.cos(angle), +math.sin(angle)],
            [-math.sin(angle), +math.cos(angle)]
        ])

        elipse = rotation @ (np.array([x, y]))
        px = np.array(elipse[0, :] + x_estimation[0, 0]).flatten()
        py = np.array(elipse[1, :] + x_estimation[1, 0]).flatten()

        if landmarks_data == (0, 0):
            axes.plot(px, py, line)
        else:
            landmarks_label = f'{area:2.4f}m2: Elipse Area, Estimated Position\n'
            landmarks_label += f'{landmarks_data[1]/landmarks_data[0]:2.4f}m2: Elipse Area, '
            landmarks_label += f'Landmark Average [{landmarks_data[0]:02d}]'

            axes.plot(
                px,
                py,
                line,
                label = landmarks_label
            )

        return area


    def generate_landmarks(
            origin: tuple, r_min: float, r_max:float, n: int
        ) -> np.ndarray[float]:
        """
        Generates random points in polar coordinates.

        Args:
            origin (tuple) : (x, y) coordinates of circle's center.
            r_min (float) : minimum radius of the circle.
            r_max (float) : maximum radius of the circle.
            n (int) : number of points to generate.
            full (bool) : consider full circle? Default is True.
        """
        angles = np.random.uniform(0, 2 * np.pi, n)
        distances = np.random.uniform(r_min, r_max, n)

        x = origin[0] + distances * np.cos(angles)
        y = origin[1] + distances * np.sin(angles)

        for i, j in zip(x, y):
            print(f'[{i:+2.1f}, {j:+2.1f}],')

        return np.column_stack((x, y))



class Simulation:
    def __init__(
            self,
            dt,
            landmarks,
            simulation_duration,
            observation_range,
            x_true,
            x_odometry,
            x_estimation,
            P_estimation,
            Q_true,
            R_true,
        ) -> None:
        self.dt = dt
        self.landmarks = landmarks
        self.simulation_duration = simulation_duration

        self.observation_range = observation_range

        self.x_true = x_true
        self.x_odometry = x_odometry
        self.Q_true = Q_true
        self.R_true = R_true

        self.history_x_true = x_true
        self.history_x_odometry = x_odometry
        self.history_x_estimation = x_estimation

        self.history_x_error = np.abs(x_estimation - x_true)
        self.history_x_covariance = np.sqrt(
            np.diag(P_estimation[0:S_S, 0:S_S]).reshape(3,1)
        )

        self.n_steps = int(np.round(simulation_duration/dt))
        self.history_time = [0]
        self.landmarks_true_id = []
        self.plot_axes = self.plot_init()



    def get_observation(self, v, w, landmarks):
        """
        Generate noisy control and get_observation and update true position and dead reckoning
        """
        u_true = self.get_robot_control(v, w)

        # add noise to gps x-y
        y = np.zeros((0, 3))

        for i in range(len(landmarks[:, 0])):
            dx = landmarks[i, 0] - self.x_true[0, 0]
            dy = landmarks[i, 1] - self.x_true[1, 0]
            angle = Utils.convert_angle(math.atan2(dy, dx) - self.x_true[2, 0])

            d = math.hypot(dx, dy)
            if d <= self.observation_range:
                d_tilde = d + np.random.randn() * self.R_true[0, 0] ** 0.5  # add noise
                d_tilde = max(d_tilde,0)
                angle_tilde = angle + np.random.randn() * self.R_true[1, 1] ** 0.5  # add noise

                yi = np.array([d_tilde, angle_tilde, i])
                y = np.vstack((y, yi))

        # add noise to input
        u_tilde = np.array([[
            u_true[0, 0] + np.random.randn() * self.Q_true[0, 0] ** 0.5,
            u_true[1, 0] + np.random.randn() * self.Q_true[1, 1] ** 0.5
        ]]).T

        self.x_odometry = Utils.compute_motion(self.x_odometry, u_tilde, self.dt)

        return y, u_tilde


    def get_robot_control(self, v: float = 1.0, w: float = 0.1) -> np.ndarray[float]:
        """
        Return robot true control command as np.ndarray.

        Note: by default a circular trajectory is generated.

        Args:
            v (float) : tangencial velocity in m/s. Default value is 1.0 m/s.
            w (float) : angular velocity in rad/s. Default value is 0.1 rad/s.
        """
        u = np.array([[v, w]]).T

        return u


    def plot_init(self) -> list:
        """Return simulation plot axis."""
        _, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(16, 8))

        ax3 = plt.subplot(3, 2, 2)
        ax4 = plt.subplot(3, 2, 4)
        ax5 = plt.subplot(3, 2, 6)

        return ax1, ax2, ax3, ax4, ax5


    def plot_iteration(
        self,
        x_estimation: np.ndarray[float],
        P_estimation: np.ndarray[float],
        show_legend: bool = False
    ) -> None:
        """
        Update simulation plot.

        Args:
            x_estimation (np.ndarray[float]) : system state estimation at instant k.
            P_estimation (np.ndarray[float]) : system state estimation covariance at instant k.
            show_legend (bool) : add legend to plot? Default value is False.
        """
        # Unfold axis
        ax1, ax2, ax3, ax4, ax5 = self.plot_axes


        # Initialize plot
        x_max, x_min = +25, -25
        y_max, y_min = +40, -10
        x_ticks = [i for i in range(x_min, x_max+1, 5)]
        y_ticks = [i for i in range(y_min, y_max+1, 5)]

        ax1.cla()
        ax1.grid(True)
        ax1.set_title('Cartesian Coordinates')
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('y [m]')
        ax1.set_xticks(x_ticks)
        ax1.set_yticks(y_ticks)
        ax1.axis([x_min, x_max, y_min, y_max])

        ax2.set_xticks([])

        ax3.set_title('Extended Kalman Filter SLAM')

        ax5.set_xlabel('time [s]')


        # Plot true landmark and trajectory
        ax1.plot(self.landmarks[:, 0], self.landmarks[:, 1], "*k")
        ax1.plot(
            self.history_x_true[0, :],
            self.history_x_true[1, :],
            "-k",
            label="True Trajectory"
        )

        # Estimated odometry trajectory
        ax1.plot(
            self.history_x_odometry[0, :],
            self.history_x_odometry[1, :],
            "-g",
            label="Odometry Measure"
        )

        # Estimated EKF trajectory and pose covariance
        ax1.plot(
            self.history_x_estimation[0, :],
            self.history_x_estimation[1, :],
            "-y",
            label="Extended Kalman Filter Trajectory"
        )
        ax1.plot(x_estimation[0], x_estimation[1], ".r")
        Utils.plot_covariance_ellipse(
            x_estimation[0: S_S],
            P_estimation[0: S_S, 0: S_S], 
            "--r",
            ax1,
            (0,0)
        )


        # Covariance ellipses
        landmarks_area = 0
        landmark_count = Utils.count_landmarks(x_estimation)
        for i in range(landmark_count):
            landmark_i = S_S + i * 2
            ax1.plot(x_estimation[landmark_i], x_estimation[landmark_i + 1], "xr")

            landmarks_area += Utils.plot_covariance_ellipse(
                x_estimation[landmark_i : landmark_i + 2],
                P_estimation[landmark_i : landmark_i + 2, landmark_i : landmark_i + 2],
                "--r",
                ax1,
                (0, 0)
            )

        Utils.plot_covariance_ellipse(
            x_estimation[0: S_S],
            P_estimation[0: S_S, 0: S_S],
            "--r",
            ax1,
            (landmark_count, landmarks_area) if show_legend else (0, 0)
        )
        ax1.legend(loc='upper left')


        # Erros and covariances plot
        error_axes = [ax3, ax4, ax5]
        ylabels = ['x [m]', 'y [m]', r"$\theta$ [rad]"]
        x_ticks = [i for i in range(0, self.simulation_duration*10+1, 50)]

        for i, error_axis in enumerate(error_axes):
            # plot errors curves
            if show_legend:
                rms_err = Utils.compute_rms_error(1 * self.history_x_error[i, :])
                rms_cov = Utils.compute_rms_error(3 * self.history_x_covariance[i, :])

                label_err = f'{rms_err:2.4f} error'
                label_cov = f'{rms_cov:2.4f} 3$\sigma$ covariance'

                error_axis.plot(+1.0 * self.history_x_error[i, :],'b', label=label_err)
                error_axis.plot(+3.0 * self.history_x_covariance[i, :],'r', label=label_cov)
            else:
                error_axis.plot(+1.0 * self.history_x_error[i, :],'b')
                error_axis.plot(+3.0 * self.history_x_covariance[i, :],'r')

            error_axis.plot(-3.0 * self.history_x_covariance[i, :],'r')

            error_axis.fill_between(
                self.history_time,
                +3.0 * self.history_x_covariance[i, :],
                -3.0 * self.history_x_covariance[i, :],
                color='gray',
                alpha=0.75
            )

            error_axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            error_axis.set_ylabel(ylabels[i])
            error_axis.set_xlim(0, self.simulation_duration)
            error_axis.set_xticks(x_ticks)

            if error_axis != ax5:
                error_axis.set_xticklabels(['' for _ in x_ticks])
            error_axis.grid(True)

            if show_legend: error_axis.legend(loc='upper right')

        plt.tight_layout()
        plt.pause(0.0001)


    def simulate_world(self, v: float = 1.0, w: float = 0.1) -> None:
        """
        Simulate system.

        Args:
            v (float) : tangencial velocity in m/s. Default value is 1.0 m/s.
            w (float) : angular velocity in rad/s. Default value is 0.1 rad/s.
        """
        self.x_true = Utils.compute_motion(
            self.x_true, self.get_robot_control(v, w), self.dt
        )
        self.x_true[2, 0] = Utils.convert_angle(self.x_true[2, 0])


    def update_history(
            self, k: int, x_estimation: np.ndarray[float], P_estimation: np.ndarray[float]
        ) -> None:
        """
        Update simulation history.

        Args:
            k (int) : simulation instant.
            x_estimation (np.ndarray[float]) : system state estimation at instant k.
            P_estimation (np.ndarray[float]) : system state estimation covariance at instant k.
        """
        self.history_x_true = np.hstack((self.history_x_true, self.x_true))
        self.history_x_odometry = np.hstack((self.history_x_odometry, self.x_odometry))
        self.history_x_estimation = np.hstack((
            self.history_x_estimation, x_estimation[0:S_S]
        ))

        error = x_estimation[0:S_S] - self.x_true
        error[2, 0] = Utils.convert_angle(error[2, 0])

        self.history_x_error = np.hstack((self.history_x_error, error))
        self.history_x_covariance = np.hstack((
            self.history_x_covariance, np.sqrt(
                np.diag(P_estimation[0:S_S, 0:S_S]).reshape(3,1)
            )
        ))
        self.history_time.append(10 * k * self.dt)



def execution(
        landmarks: np.ndarray[float],
        landmarks_known: bool = True,
        undelayed_method: bool = True,
        v: float = 1.0,
        w: float = 0.1,
        dt: float = 0.1,
        observation_range: int = 10,
        P_constant: int = 1,
        Q_constant: int = 6,
        R_constant: int = 2,
        save: bool = False,
        show: bool = True,
    ) -> None:
    """
    Execute an Particles Filter simulation for SLAM.

    Args:
        landmarks (np.ndarray[float]) : array of landmarks coordinates.
        landmarks_known (bool) : use landmarks true positions? Default value is True.
        dt (int) : prediction interval in seconds. Default value is 1.
        observation_range (int) : robot observation range. Default value is 10.
        P_constant (int) : state noise covariance matrix P constant. Default value is 1.
        Q_constant (int) : process noise covariance matrix Q constant. Default value is 1.
        R_constant (int) : measure noise covariance matrix R constant. Default value is 1.
        save (bool) : save result? Default value is False.
        show (bool) : show result? Default value is True.
    """
    # Define Kalman covariance errors robot movements
    P_true = np.diag([0.01, 0.01, 0.0001])
    Q_true = np.diag([0.1, np.deg2rad(1)]) ** 2     # input noise
    R_true = np.diag([0.1, np.deg2rad(5)]) ** 2     # measurement noise

    P_estimation = P_constant * P_true
    Q_estimation = Q_constant * Q_true  # estimated input noise
    R_estimation = R_constant * R_true  # estimated measurement noise


    # Simulation initial conditions
    simulation_duration = 80    # simulation time [s]

    x_true = np.zeros((S_S, 1))
    x_initial = x_true
    x_odometry = x_true
    x_estimation = x_true

    # Simulation creation
    simulation = Simulation(
        dt,
        landmarks,
        simulation_duration,
        observation_range,
        x_true,
        x_odometry,
        x_estimation,
        P_estimation,
        Q_true,
        R_true,
    )


    # Simulation execution
    for k in range(1, simulation.n_steps):
        simulation.simulate_world(v, w)
        z, u_tilde = simulation.get_observation(v, w, landmarks)

        x_estimation, P_estimation = EKF_SLAM.compute_iteration(
            z,
            u_tilde,
            x_estimation,
            P_estimation,
            Q_estimation,
            R_estimation,
            simulation.dt,
            landmarks_known,
            undelayed_method,
            simulation.landmarks_true_id,
        )

        simulation.update_history(k, x_estimation, P_estimation)

        # Verify loop closure
        if Utils.is_initial_position(x_initial, x_estimation):
            for error_axes in simulation.plot_axes[2:]:
                error_axes.axvline(x=k, color='orange', linestyle='--', linewidth=1)

        # Simulation plot
        if show and (k % 10 == 0):
            simulation.plot_iteration(x_estimation, P_estimation)

    simulation.plot_iteration(x_estimation, P_estimation, show_legend=True)


    # Simulation save / show
    file_name = f'EKF_SLAM_{dt}_{v}_{w}_'
    file_name += f'{landmarks_known}_{landmarks.shape[0]}_{observation_range}_'
    file_name += f'{P_constant}_{Q_constant}_{R_constant}.png'

    file_path = os.path.join(os.path.abspath(os.getcwd()), '../outputs', file_name)

    plt.suptitle(file_name)
    if save: plt.savefig(file_path, dpi=300)
    if show: plt.show()


# KNOWN_DATA_ASSOCIATION = 0  # Whether we use the true landmarks id or not

def main():
    # Utils.generate_landmarks((0, 0), 0, 10, 10)

    default = {
        'v': 1.0,
        'w': 0.1,
        'landmarks': np.array([
            [+00.0, +05.0], [+11.0, +01.0], [+03.0, +15.0], [-05.0, +20.0]
        ])
    }

    # short loop and a dense map
    scenario_0 = {
        'v': 1.5,
        'w': 0.2,
        'landmarks': np.array([
            [+03.0, +10.6], [+13.2, +04.7], [-00.6, +16.1], [+08.2, +01.4], [-12.4, +19.6],
            [-08.7, +21.9], [-03.2, +08.5], [+08.7, -06.2], [+10.9, +18.1], [-10.7, +04.5],
            [-07.2, +14.8], [-06.9, +09.8], [+07.6, +15.5], [+08.9, -03.3], [+03.9, -00.8],
            [-07.1, +17.0], [+02.2, +23.7], [+01.9, -03.4], [-11.3, -00.2], [+03.9, +12.1],
            [+08.8, -04.2], [+01.0, +18.7], [+06.2, +15.6], [-11.4, +00.3], [-07.0, +04.0],
        ])
    }

    # long loop and a dense map
    scenario_1 = {
        'v': 1.5,
        'w': 0.1,
        'landmarks': np.array([
            [+16.3, +31.7], [+05.6, +13.8], [-01.5, +37.2], [+16.0, +02.9], [-09.4, +24.2],
            [-06.1, +25.2], [-08.0, +17.5], [+06.9, +04.1], [+12.8, +27.5], [-10.1, +12.2], 
            [-09.9, +25.2], [-21.2, +22.2], [+08.9, +24.5], [+11.0, +01.7], [+03.1, +08.3], 
            [-11.4, +30.3], [+01.8, +28.2], [+02.0, +03.3], [-14.0, +05.5], [+05.8, +21.8], 
            [+04.2, +09.4], [+01.8, +34.6], [+03.2, +19.2], [-12.4, +07.2], [-06.2, +11.9], 
            [+13.9, +20.6], [-18.4, +03.3], [+00.7, +37.9], [-10.6, +31.6], [+20.1, +13.1],
        ])
    }

    # long loop and a sparse map
    scenario_2 = {
        'v': 1.5,
        'w': 0.1,
        'landmarks': np.array([
            [+2.6, +2.7], [+4.4, -0.9], [-0.1, +1.3], [+6.9, -5.2], [-5.9, +5.7],
            [-1.8, +3.0], [-2.2, +0.7], [+4.2, -6.6], [+4.3, +4.2], [-1.3, -0.4], 
        ])
    }

    scenario_3 = {
        'v': 1.5,
        'w': 0.1,
        'landmarks': np.array([
            [0.0, +7.5]
        ])
    }

    # return
    for scenario in [scenario_3]:
        for known in [False]:
            for Q_constant in [6]:
                execution(
                    landmarks=scenario['landmarks'],
                    landmarks_known=known,
                    undelayed_method=True,
                    v=scenario['v'],
                    w=scenario['w'],
                    Q_constant=Q_constant,
                    save=False,
                    show=True
                )



if __name__ == '__main__':
    main()
