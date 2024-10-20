"""CSC_5RO12_TA_TP2, Navegation pour la Robotique algorithm."""

from dataclasses import dataclass
from math import sin, cos, atan2, pi, sqrt
from matplotlib.ticker import FormatStrFormatter

import matplotlib.pyplot as plt
import numpy as np
import os

seed = 123456
np.random.seed(seed)
try:
    os.makedirs("../outputs")
except:
    pass



@dataclass
class EKF:
    """"Extended Kalman Filter equations."""

    def motion_model_prediction(
            x: np.ndarray[float], u: np.ndarray[float], dt: float
        ) -> np.ndarray[float]:
        """
        Returns motion model prediction as np.ndarray[float] of system state x at instant k.

        Args:
            x (np.ndarray[float]) : system state at instant k-1.
            u (np.ndarray[float]) : control input, or odometry measurement, at instant k.
            dt (float) : simulation time step in seconds.
        """
        x_k, y_k, theta_k = x[0, 0], x[1, 0], x[2, 0]
        vx_k, vy_k, w_k = u[0, 0], u[1, 0], u[2, 0]

        x_prediction = np.array([
            [x_k + (vx_k * np.cos(theta_k) - vy_k * np.sin(theta_k)) * dt],
            [y_k + (vx_k * np.sin(theta_k) + vy_k * np.cos(theta_k)) * dt],
            [theta_k + w_k * dt],
        ])
        x_prediction[2, 0] = Utils.convert_angle(x_prediction[2, 0])

        return x_prediction


    def observation_model_prediction(
            x: np.ndarray[float], i: int, landmarks: np.ndarray[float]
        ) -> np.ndarray[float]:
        """
        Returns observation model prediction as np.ndarray[float] of system command y at instant k.

        Args:
            x (np.ndarray[float]) : system state at instant k-1.
            i (int) : observed landmark index.
            landmarks (np.ndarray[float]) : coordinates x and y of all landmarks.
        """
        x_k, y_k, theta_k = x[0, 0], x[1, 0], x[2, 0]
        x_i, y_i = landmarks[0, i], landmarks[1, i]

        h = np.array([
            [np.sqrt((x_i - x_k)**2 + (y_i - y_k)**2)],
            [np.arctan2((y_i - y_k), (x_i - x_k)) - theta_k],
        ])
        h[1, 0] = Utils.convert_angle(h[1, 0])

        return h


    def F(x: np.ndarray[float], u: np.ndarray[float], dt: float) -> np.ndarray[float]:
        """
        Return motion model state jacobian matrix F(x) as np.ndarray[float].

        Note: jacobian with respect of system state.

        Args:
            x (np.ndarray[float]) : system state at instant k-1.
            u (np.ndarray[float]) : control input, or odometry measurement, at instant k.
            dt (float) : simulation time step in seconds.
        """
        _, _, theta_k = x[0, 0], x[1, 0], x[2, 0]
        vx_k, vy_k, _ = u[0, 0], u[1, 0], u[2, 0]

        F = np.array([
            [1, 0, (-vx_k * np.sin(theta_k) - vy_k * np.cos(theta_k)) * dt],
            [0, 1, (+vx_k * np.cos(theta_k) - vy_k * np.sin(theta_k)) * dt],
            [0, 0, 1]
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
        _, _, theta_k = x[0, 0], x[1, 0], x[2, 0]

        G = np.array([
            [np.cos(theta_k) * dt, -np.sin(theta_k) * dt, 0],
            [np.sin(theta_k) * dt, +np.cos(theta_k) * dt, 0],
            [0, 0, dt]
        ])

        return G


    def H(x: np.ndarray[float], i: int, landmarks: np.ndarray[float]) -> np.ndarray[float]:
        """
        Return observation model jacobian matrix H(x) as np.ndarray[float].

        Args:
            x (np.ndarray[float]) : system state at instant k-1.
            i (int) : observed landmark index.
            landmarks (np.ndarray[float]) : coordinates x and y of all landmarks.
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



@dataclass
class Utils:
    """Utils functions needed for simulation."""

    def plot_covariance_ellipse(
            x_estimation: np.ndarray[float], P_estimation: np.ndarray[float], line: str, axes
        ) -> None:
        """
        Plot motion estimation convariance matrix as an ellipse.

        Args:
            x_estimation (np.ndarray[float]) : system state estimation at instant k.
            P_estimation (np.ndarray[float]) : estimation covariance at instant k.
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
            print('Pb with P_xy :\n', P_xy)
            exit()

        a = sqrt(eigen_values[index_big])
        b = sqrt(eigen_values[index_small])
        area = pi * a * b

        circle = np.arange(0, 2 * pi + 0.1, 0.1)
        x = [3 * a * cos(angle) for angle in circle]
        y = [3 * b * sin(angle) for angle in circle]

        angle = atan2(eigen_vectors[index_big, 1], eigen_vectors[index_big, 0])
        rotation = np.array([[+cos(angle), +sin(angle)],
                             [-sin(angle), +cos(angle)]])

        ellipse = rotation @ (np.array([x, y]))
        px = np.array(ellipse[0, :] + x_estimation[0, 0]).flatten()
        py = np.array(ellipse[1, :] + x_estimation[1, 0]).flatten()

        axes.plot(px, py, line, label=f'Ellipse Area: {area:2.4f} m$^2$')
 

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

        x_k, y_k, theta_k = x[0, 0], x[1, 0], x[2, 0]
        vx_k, vy_k, w_k = u[0, 0], u[1, 0], u[2, 0]

        x_calculated = np.array([
            [x_k + (vx_k * np.cos(theta_k) - vy_k * np.sin(theta_k)) * dt],
            [y_k + (vx_k * np.sin(theta_k) + vy_k * np.cos(theta_k)) * dt],
            [theta_k + dt * w_k],
        ])
        x_calculated[2, 0] = Utils.convert_angle(x_calculated[2, 0])

        return x_calculated


    def compute_rms_error(arr: np.ndarray[float]) -> float:
        """
        Return RMS error of an np.ndarray as a float.

        Args:
            arr (np.ndarray) : array to analyze.
        """
        return np.sqrt(np.mean(arr**2))


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



class Simulation:
    def __init__(
            self,
            simulation_duration,
            dt_measurement,
            dt_prediction,
            landmarks,
            x_estimation,
            x_odometry,
            x_true,
            P_estimation,
            Q_true,
            R_true,
        ):
        self.simulation_duration = simulation_duration
        self.landmarks = landmarks
        self.n_steps = int(np.round(simulation_duration/dt_prediction))
        self.dt_measurement = dt_measurement
        self.dt_prediction = dt_prediction

        self.x_odometry = x_odometry
        self.x_true = x_true
        self.Q_true = Q_true
        self.R_true = R_true

        self.history_x_estimation = x_estimation
        self.history_x_odometry = x_odometry
        self.history_x_true = x_true

        self.history_x_error = np.abs(x_estimation - x_true)  # pose error
        self.history_x_variance = np.sqrt(np.diag(P_estimation).reshape(3, 1))  # state std dev
        self.history_time = [0]


    def get_observation(
            self, k: int, black_out: bool = False
        ) -> list[np.ndarray, int] | list[None, None]:
        """
        Return a noisy observation of a random landmark at instant k.

        Args:
            k (int) : interation step.
        """
        np.random.seed(seed*3 + k)  # Ensuring random repexility for k

        if k * self.dt_prediction % self.dt_measurement == 0:
            if black_out:
                valid_measurement = False if 2500 < k < 3500 else True
            else:
                valid_measurement = True

            if not valid_measurement:
                return None, None
            else:
                landmark_index = np.random.randint(0, self.landmarks.shape[1] - 1)
                z_noise = np.sqrt(self.R_true) @ np.random.randn(2)
                z_noise = np.array([z_noise]).T

                z = EKF.observation_model_prediction(self.x_true, landmark_index, self.landmarks)
                z = z + z_noise
                z[1, 0] = Utils.convert_angle(z[1, 0])

                return z, landmark_index

        return None, None


    def get_odometry(self, k: int) -> list[np.ndarray]:
        """
        Return a noisy odometry and command measurements at instant k as np.ndarrays.

        Args:
            k (int) : interation step.
        """
        np.random.seed(seed*2 + k) # Ensuring random repexility

        u_noise = np.sqrt(self.Q_true) @ np.random.randn(3)
        u_noise = np.array([u_noise]).T
        u = self.get_robot_control(k) + self.dt_prediction*u_noise

        x = Utils.compute_motion(self.x_odometry, self.get_robot_control(k), self.dt_prediction)
        x = Utils.compute_motion(x, u_noise, self.dt_prediction)
        self.x_odometry = x

        return x, u


    def plot(
            self,
            file_name: str,
            x_estimation: np.ndarray,
            P_estimation: np.ndarray,
            show: bool = True,
            save: bool = False
        ) -> None:
        """"
        Plot simulation results.
        """
        _, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(16, 8))
        ax3 = plt.subplot(3, 2, 2)
        ax4 = plt.subplot(3, 2, 4)
        ax5 = plt.subplot(3, 2, 6)
        ax1.cla()

        times = np.stack(self.history_time)

        # True trajectory and landmark
        ax1.plot(self.landmarks[0, :], self.landmarks[1, :], "*k")
        ax1.plot(
            self.history_x_true[0, :],
            self.history_x_true[1, :],
            "-k",
            label="Trajectory"
        )

        # Estimated odometry trajectory
        ax1.plot(
            self.history_x_odometry[0, :],
            self.history_x_odometry[1, :],
            "-g",
            label="Odometry"
        )

        # Estimated EKF trajectory and pose covariance
        ax1.plot(
            self.history_x_estimation[0, :],
            self.history_x_estimation[1, :],
            "-y",
            label="Extended Kalman Filter"
        )
        ax1.plot(x_estimation[0], x_estimation[1], ".r")
        Utils.plot_covariance_ellipse(x_estimation, P_estimation,  "--r", ax1)

        ax1.grid(True)
        ax1.axis([-100, 100, -100, 100])
        ax1.set_title('Cartesian Coordinates')
        ax1.set_ylabel('y [m]')
        ax1.set_xlabel('x [m]')
        ax1.legend(loc='upper left')

        # add common error markers
        ax2.set_xticks([])

        # plot errors curves
        x_ticks = [i for i in range(0, 6001, 500)]

        rms_x_err = Utils.compute_rms_error(self.history_x_error[0, :])
        rms_y_err = Utils.compute_rms_error(self.history_x_error[1, :])
        rms_theta_err = Utils.compute_rms_error(self.history_x_error[2, :])

        rms_x_cov = Utils.compute_rms_error(3*self.history_x_variance[0, :])
        rms_y_cov = Utils.compute_rms_error(3*self.history_x_variance[1, :])
        rms_theta_cov = Utils.compute_rms_error(3*self.history_x_variance[2, :])

        label_x_err = f'{rms_x_err:2.4f} error'
        label_y_err = f'{rms_y_err:2.4f} error'
        label_theta_err = f'{rms_theta_err:2.4f} error'

        label_x_cov = f'{rms_x_cov:2.4f} 3$\sigma$ covariance'
        label_y_cov = f'{rms_y_cov:2.4f} 3$\sigma$ covariance'
        label_theta_cov = f'{rms_theta_cov:2.4f} 3$\sigma$ covariance'

        ax3.plot(times, self.history_x_error[0, :], 'b', label=label_x_err)
        ax3.plot(times, +3.0 * self.history_x_variance[0, :], 'r', label=label_x_cov)
        ax3.plot(times, -3.0 * self.history_x_variance[0, :], 'r')
        ax3.set_title('Extended Kalman Filter')
        ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax3.set_ylabel('x [m]')
        ax3.set_xlim(0, 6000)
        ax3.set_xticks(x_ticks)
        ax3.set_xticklabels(['' for _ in x_ticks])
        ax3.legend(loc='upper right')
        ax3.grid(True)

        ax4.plot(times, self.history_x_error[1, :], 'b', label=label_y_err)
        ax4.plot(times, +3.0 * self.history_x_variance[1, :], 'r', label=label_y_cov)
        ax4.plot(times, -3.0 * self.history_x_variance[1, :], 'r')
        ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax4.set_ylabel('y [m]')
        ax4.set_xlim(0, 6000)
        ax4.set_xticks(x_ticks)
        ax4.set_xticklabels(['' for _ in x_ticks])
        ax4.legend(loc='upper right')
        ax4.grid(True)

        ax5.plot(times, self.history_x_error[2, :], 'b', label=label_theta_err)
        ax5.plot(times, +3.0 * self.history_x_variance[2, :], 'r', label=label_theta_cov)
        ax5.plot(times, -3.0 * self.history_x_variance[2, :], 'r')
        ax5.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax5.set_ylabel(r"$\theta$ [rad]")
        ax5.set_xlabel('time [s]')
        ax5.set_xlim(0, 6000)
        ax5.set_xticks(x_ticks)
        ax5.legend(loc='upper right')
        ax5.grid(True)

        file_path = os.path.join(os.path.abspath(os.getcwd()), 'outputs', file_name)

        plt.suptitle(file_name)
        plt.tight_layout()
        if save: plt.savefig(file_path, dpi=300)
        if show: plt.show()


    def get_robot_control(self, k: int) -> np.ndarray[float]:
        """
        Return robot true control command at instant k as np.ndarray.

        Note: by default a sinousal trajectory is generated.

        Args:
            k (int) : interation step.
        """
        u = np.array([[0, 0.025,  0.1*np.pi / 180 * sin(3*np.pi * k / self.n_steps)]]).T

        return u


    def simulate_world(self, k: int) -> None:
        """
        Simulate system at instant k.

        Args:
            k (int) : interation step.
        """
        self.x_true = Utils.compute_motion(
            self.x_true, self.get_robot_control(k), self.dt_prediction
        )
        self.x_true[2, 0] = Utils.convert_angle(self.x_true[2, 0])


    def update_history(
            self, k: int, x_estimation: np.ndarray[float], P_estimation: np.ndarray[float]
        ) -> None:
        """
        Update simulation history.

        Args:
            x_estimation (np.ndarray[float]) : system state estimation at instant k.
            P_estimation (np.ndarray[float]) : estimation covariance at instant k.
        """
        self.history_x_true = np.hstack((self.history_x_true, self.x_true))
        self.history_x_odometry = np.hstack((self.history_x_odometry, self.x_odometry))
        self.history_x_estimation = np.hstack((self.history_x_estimation, x_estimation))

        error = x_estimation - self.x_true
        error[2, 0] = Utils.convert_angle(error[2, 0])

        self.history_x_error = np.hstack((self.history_x_error, error))
        self.history_x_variance = np.hstack((
            self.history_x_variance, np.sqrt(np.diag(P_estimation).reshape(3, 1))
        ))
        self.history_time.append(k*self.dt_prediction)



def execution(
    dt_measurement: int = 1,
    dt_prediction: int = 1,
    n_landmarks: int = 30,
    P_constant: int = 1,
    Q_constant: int = 1,
    R_constant: int = 1,
    black_out: bool = False,
    range_only: bool = False,
    angle_only: bool = False,
    show_result: bool = True,
    save_result: bool = True
    ) -> None:
    """
    Execute an Extended Kalman Filter simulation.

    Args:
        dt_measurement (int) : measurement interval in seconds. Default values is 1.
        dt_prediction (int) : prediction interval in seconds. Default values is 1.
        n_landmarks (int) : number of landmarks simulated. Default values is 30.
        P_constant (int) : state covariance matrix P constant. Default values is 1.
        Q_constant (int) : process noise covariance matrix Q constant. Default values is 1.
        R_constant (int) : measure noise covariance matrix R constant. Default values is 1.
        black_out (bool) : no measures between 2500 s and 3000s? Default values is False.
        range_only (bool) : only range measures available? Default values is False.
        angle_only (bool) : only angle measures available? Default values is False.
        show_result (bool) : show result? Default value is True.
        save_result (bool) : save result? Default value is True.
    """
    # Define Kalman covariance errors robot movements
    P_true = np.diag([1, 1, (1*pi/180)**2])
    Q_true = np.diag([0.01, 0.01, 1*pi/180]) ** 2
    R_true = np.diag([3.0, 3*pi/180]) ** 2

    P_estimation = P_constant * np.diag([1, 1, 1]) @ P_true
    Q_estimation = Q_constant * np.diag([1, 1, 1]) @ Q_true
    R_estimation = R_constant * np.diag([1, 1]) @ R_true

    if range_only: R_estimation = R_estimation[0:1, 0:1]    # exclude angle measurement
    if angle_only: R_estimation = R_estimation[1:, 1:]      # exclude range measurement


    # Simulation initial conditions
    simulation_duration = 6000  # simulation duration [s]
    landmarks = 140*(np.random.rand(2, n_landmarks) - 1/2)

    x_true = np.array([[1, -40, -pi/2]]).T
    x_odometry = x_true
    x_estimation = x_true

    # Simulation environment
    simulation = Simulation(
        simulation_duration,
        dt_measurement,
        dt_prediction,
        landmarks,
        x_estimation,
        x_odometry,
        x_true,
        P_estimation,
        Q_true,
        R_true,
    )


    # Loop simulation environment
    for k in range(1, simulation.n_steps):
        simulation.simulate_world(k)    # Simulate robot motion
        x_odometry, u_tilde = simulation.get_odometry(k)
        z, landmark_index = simulation.get_observation(k, black_out=black_out)

        # Kalman Prediction
        x_prediction = EKF.motion_model_prediction(x_estimation, u_tilde, simulation.dt_prediction)

        F = EKF.F(x_prediction, u_tilde, simulation.dt_prediction)
        G = EKF.G(x_prediction, u_tilde, simulation.dt_prediction)
        P_prediction = F @ P_estimation @ F.T + G @ Q_estimation @ G.T

        # Kalman Correction
        if z is not None:
            # Observation available
            h = EKF.observation_model_prediction(x_prediction, landmark_index, simulation.landmarks)

            innovation = z - h
            innovation[1, 0] = Utils.convert_angle(innovation[1, 0])
            if range_only: innovation = innovation[0:1, :]  # exclude angle measurement
            if angle_only: innovation = innovation[1:, :]   # exclude range measurement


            H = EKF.H(x_prediction, landmark_index, simulation.landmarks)
            if range_only: H = H[0:1, :]    # exclude angle measurement
            if angle_only: H = H[1:, :]     # exclude range measurement

            S = R_estimation + H @ P_prediction @ H.T
            K = P_prediction @ H.T @ np.linalg.inv(S)

            # Kalman Update
            x_estimation = x_prediction + K @ innovation
            x_estimation[2, 0] = Utils.convert_angle(x_estimation[2, 0])

            P_estimation = (np.eye(K.shape[0]) - K @ H) @ P_prediction
            P_estimation = 0.5 * (P_estimation + P_estimation.T)  # symetry matrix
        else:
            # Observation unavailable
            x_estimation = x_prediction
            P_estimation = P_prediction

        # Update data history
        simulation.update_history(k, x_estimation, P_estimation)

    file_name = f'EKF_{dt_measurement}_{dt_prediction}_'
    file_name += f'{n_landmarks}_'
    file_name += f'{Q_constant}_{R_constant}_{P_constant}_'
    file_name += f'{black_out}_{range_only}_{angle_only}.png'

    simulation.plot(file_name, x_estimation, P_estimation, show_result, save_result)



def main():
    # arr = [1]
    arr = [5, 10, 100, 150]
    # arr = [10, 25, 100, 250]

    for var in arr:
        execution(n_landmarks=var, angle_only=True, show_result=False, save_result=True)



if __name__ == "__main__":
    main()
