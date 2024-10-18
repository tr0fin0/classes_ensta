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
            x: np.ndarray[float], i: int, references: np.ndarray[float]
        ) -> np.ndarray[float]:
        """
        Returns observation model prediction as np.ndarray[float] of system command y at instant k.

        Args:
            x (np.ndarray[float]) : system state at instant k-1.
            i (int) : observed landmark index.
            references (np.ndarray[float]) : coordinates x and y of all references.
        """
        x_k, y_k, theta_k = x[0, 0], x[1, 0], x[2, 0]
        x_i, y_i = references[0, i], references[1, i]

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


    def H(x: np.ndarray[float], i: int, references: np.ndarray[float]) -> np.ndarray[float]:
        """
        Return observation model jacobian matrix H(x) as np.ndarray[float].

        Args:
            x (np.ndarray[float]) : system state at instant k-1.
            i (int) : observed landmark index.
            references (np.ndarray[float]) : coordinates x and y of all references.
        """
        x_k, y_k = x[0, 0], x[1, 0]
        x_p, y_p = references[0, i], references[1, i]

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
        axes.plot(px, py, line)
 

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
            references,
            x_estimation,
            x_odometry,
            x_true,
            P_estimation,
            Q_true,
            R_true,
        ):
        self.simulation_duration = simulation_duration
        self.references = references
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


    def get_observation(self, k: int) -> list[np.ndarray, int] | list[None, None]:
        """
        Return a noisy observation of a random landmark at instant k.

        Args:
            k (int) : interation step.
        """
        np.random.seed(seed*3 + k)  # Ensuring random repexility for k

        if k * self.dt_prediction % self.dt_measurement == 0:
            valid_measurement = True

            if not valid_measurement:
                return None, None
            else:
                landmark_index = np.random.randint(0, self.references.shape[1] - 1)
                z_noise = np.sqrt(self.R_true) @ np.random.randn(2)
                z_noise = np.array([z_noise]).T

                z = EKF.observation_model_prediction(self.x_true, landmark_index, self.references)
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



def main():
    # Define ploting characteristics
    show_animation = True
    save_iterations = False

    _, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(16, 8))
    ax3 = plt.subplot(3, 2, 2)
    ax4 = plt.subplot(3, 2, 4)
    ax5 = plt.subplot(3, 2, 6)


    # Define Kalman covariance errors robot movements
    P_constant = 1
    Q_constant = 1
    R_constant = 1

    Q_true = np.diag([0.01, 0.01, 1*pi/180]) ** 2
    R_true = np.diag([3.0, 3*pi/180]) ** 2

    P_estimation = P_constant * np.diag([1, 1, (1*pi/180)**2])
    Q_estimation = Q_constant * np.eye(3, 3) @ Q_true
    R_estimation = R_constant * np.eye(2, 2) @ R_true


    # Simulation characteristics
    simulation_duration = 6000  # simulation duration [s]
    dt_prediction = 1           # dynamical prediction interval [s]
    dt_measurement = 1          # measurement update interval [s]

    n_references = 30
    references = 140*(np.random.rand(2, n_references) - 1/2)

    # Simulation initial conditions
    x_true = np.array([[1, -40, -pi/2]]).T
    x_odometry = x_true
    x_estimation = x_true

    # Simulation environment
    simulation = Simulation(
        simulation_duration,
        dt_measurement,
        dt_prediction,
        references,
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
        y, landmark_index = simulation.get_observation(k)

        # Kalman Prediction
        x_prediction = EKF.motion_model_prediction(x_estimation, u_tilde, simulation.dt_prediction)

        F = EKF.F(x_prediction, u_tilde, simulation.dt_prediction)
        G = EKF.G(x_prediction, u_tilde, simulation.dt_prediction)
        P_prediction = F @ P_estimation @ F.T + G @ Q_estimation @ G.T

        # Kalman Correction
        if y is not None:
            # Observation available
            h = EKF.observation_model_prediction(x_prediction, landmark_index, simulation.references)

            innovation = y - h
            innovation[1, 0] = Utils.convert_angle(innovation[1, 0])

            H = EKF.H(x_prediction, landmark_index, simulation.references)
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


        # Plot simulation
        if show_animation and k*simulation.dt_prediction % 200 == 0:
            # stop simulation with the ESC key
            plt.gcf().canvas.mpl_connect('key_release_event',
                        lambda event: [exit(0) if event.key == 'escape' else None])

            ax1.cla()

            times = np.stack(simulation.history_time)

            # True trajectory and landmark
            ax1.plot(references[0, :], references[1, :], "*k")
            ax1.plot(
                simulation.history_x_true[0, :],
                simulation.history_x_true[1, :],
                "-k",
                label="Trajectory"
            )

            # Estimated odometry trajectory
            ax1.plot(
                simulation.history_x_odometry[0, :],
                simulation.history_x_odometry[1, :],
                "-g",
                label="Odometry"
            )

            # Estimated EKF trajectory and pose covariance
            ax1.plot(
                simulation.history_x_estimation[0, :],
                simulation.history_x_estimation[1, :],
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
            ax3.plot(times, simulation.history_x_error[0, :], 'b')
            ax3.plot(times, +3.0 * simulation.history_x_variance[0, :], 'r')
            ax3.plot(times, -3.0 * simulation.history_x_variance[0, :], 'r')
            ax3.set_title('Error [blue] and 3$\sigma$ Covariances [red]')
            ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax3.set_ylabel('x [m]')
            ax3.set_xlim(0, 6000)
            ax3.set_xticks([0, 1000, 2000, 3000, 4000, 5000, 6000])
            ax3.set_xticklabels(['', '', '', '', '', '', ''])
            ax3.grid(True)

            ax4.plot(times, simulation.history_x_error[1, :], 'b')
            ax4.plot(times, +3.0 * simulation.history_x_variance[1, :], 'r')
            ax4.plot(times, -3.0 * simulation.history_x_variance[1, :], 'r')
            ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax4.set_ylabel('y [m]')
            ax4.set_xlim(0, 6000)
            ax4.set_xticks([0, 1000, 2000, 3000, 4000, 5000, 6000])
            ax4.set_xticklabels(['', '', '', '', '', '', ''])
            ax4.grid(True)

            ax5.plot(times, simulation.history_x_error[2, :], 'b')
            ax5.plot(times, +3.0 * simulation.history_x_variance[2, :], 'r')
            ax5.plot(times, -3.0 * simulation.history_x_variance[2, :], 'r')
            ax5.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax5.set_ylabel(r"$\theta$ [rad]")
            ax5.set_xlabel('time [s]')
            ax5.set_xlim(0, 6000)
            ax5.set_xticks([0, 1000, 2000, 3000, 4000, 5000, 6000])
            ax5.grid(True)

            plt.tight_layout()

            if save_iterations: plt.savefig(r'outputs/EKF_' + str(k) + '.png')

    file_name = f'EKF_{Q_constant}_{R_constant}_{P_constant}.png'
    plt.savefig(os.path.join(os.path.abspath(os.getcwd()), '..', 'outputs', file_name), dpi=300)
    plt.show()



if __name__ == "__main__":
    main()
