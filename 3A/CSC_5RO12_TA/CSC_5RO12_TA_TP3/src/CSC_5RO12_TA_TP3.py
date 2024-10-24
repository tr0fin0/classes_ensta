""""CSC_5RO12_TA_TP3, Navegation pour la Robotique algorithm."""

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



@dataclass
class PF:
    """Particle Filter methods."""

    def motion_model_prediction(
            x: np.ndarray[float], u: np.ndarray[float], dt: float, Q_estimation: np.ndarray[float]
        ) -> np.ndarray[float]:
        """
        Return motion model of agent.

        Args:
            x (np.ndarray[float]) : system state at instant k-1.
            u (np.ndarray[float]) : control input, or odometry measurement, at instant k.
            dt (float) : simulation time step in seconds.
            Q_estimation (np.ndarray[float]) : process noise covariance matrix Q estimation.
        """
        x, y, theta = x[0, 0], x[1, 0], x[2, 0]
        vx, vy, omega = u[0, 0], u[1, 0], u[2, 0]

        w = np.random.multivariate_normal([0, 0, 0], Q_estimation)
        w_vx, w_vy, w_omega = w[0], w[1], w[2]

        x_prediction = np.array([
            [x + ((vx + w_vx) * np.cos(theta) - (vy + w_vy) * np.sin(theta)) * dt],
            [y + ((vx + w_vx) * np.sin(theta) + (vy + w_vy) * np.cos(theta)) * dt],
            [theta + (omega + w_omega) * dt]
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
        x, y, theta = x[0, 0], x[1, 0], x[2, 0]
        x_i, y_i = landmarks[0, i], landmarks[1, i]

        h = np.array([
            [np.sqrt((x_i - x)**2 + (y_i - y)**2)],
            [np.arctan2((y_i - y), (x_i - x)) - theta],
        ])
        h[1, 0] = Utils.convert_angle(h[1, 0])

        return h


    def resample_particles(
            particles: np.ndarray[float], weights: np.ndarray[float], n: int, method: str = 'low-variance'
        ) -> np.ndarray[float]:
        """
        Return resampled particles as np.ndarray[float] based on method

        Args:
            particles (np.ndarray[float]): particles states: x, y, theta.
            weights_particles (np.ndarray[float]): particle weights.
            n (int): number of particles.
            method (str) : resample method. Default is 'low-variance'.
        """
        match method.upper():
            case 'LOW-VARIANCE':
                return PF.low_variance_resample(particles, weights, n)

            case 'MULTINOMIAL':
                return PF.multinomial_resample(particles, weights, n)

            case _:
                return None


    def low_variance_resample(
            particles: np.ndarray[float], weights: np.ndarray[float], n: int
        ) -> np.ndarray[float]:
        """
        Return low-variance resampled particles based on their weights.

        Args:
            particles (np.ndarray[float]): particles states: x, y, theta.
            weights (np.ndarray[float]): particle weights.
            n (int): number of particles.
        """
        base_indices = np.arange(0.0, 1.0, 1 / n)
        random_offset = np.random.uniform(0, 1 / n)
        resampling_indices = base_indices + random_offset

        cumulative_weights = np.cumsum(weights)
        cumulative_index = 0

        resampled_indices = []
        for i in range(n):
            while resampling_indices[i] > cumulative_weights[cumulative_index]:
                cumulative_index += 1

            resampled_indices.append(cumulative_index)

        resampled_particles = particles[:, resampled_indices]
        resampled_weights = np.ones(n) / n

        return resampled_particles, resampled_weights


    def multinomial_resample(
            particles: np.ndarray[float], weights: np.ndarray[float], n: int
        ) -> np.ndarray[float]:
        """
        Return multinomial resampled particles based on their weights.

        Args:
            particles (np.ndarray[float]): particles states: x, y, theta.
            weights (np.ndarray[float]): particle weights.
            n (int): number of particles.
        """
        indices = np.random.choice(np.arange(n), size=n, p=weights)

        resampled_particles = particles[:, indices]
        resampled_weights = np.ones(n) / n

        return resampled_particles, resampled_weights


    def P(x: np.ndarray[float], particles: np.ndarray[float]) -> np.ndarray[float]:
        """
        Returns particle filter covariance matrix P estimation as np.ndarray[float].

        Args:
            x (np.ndarray[float]) : system state at instant k-1.
            particles (np.ndarray[float]): particles states: x, y, theta.
        """
        n = particles.shape[1]

        P_estimation = np.zeros((3, 3))
        for i in range(n):
            deviation = (particles[:, i:i+1] - x).reshape(-1, 1)
            deviation[2] = Utils.convert_angle(deviation[2])

            P_estimation += deviation @ deviation.T
        P_estimation /= n

        return P_estimation


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


    def weights_histogram(
            file_name: str,
            weights_particles: np.ndarray[float],
            theta_eff: float,
            save: bool = False,
            show: bool = True
        ) -> None:
        """"
        Plot weights histogram.

        Args:
            file_name (str) : output image name.
            weights_particles (np.ndarray[float]) : weights of particles at instant k.
            theta_eff (float) : resampling threshold.
            save (bool) : save result? Default value is False.
            show (bool) : show result? Default value is True.
        """
        fig, ax1 = plt.subplots(figsize=(16, 8))

        # Add histogram
        bin_count = 20
        bin_edges = np.linspace(0, 0.025, bin_count + 1)

        counts, bins, _ = ax1.hist(
            weights_particles,
            bins=bin_edges,
            alpha=0.75,
            label=f'$\\theta$ eff = {theta_eff:.2f}',
            edgecolor='black'
        )

        ax1.set_xlabel('Particles Weights')
        ax1.set_ylabel('Frequency (%)')

        total_particles = weights_particles.shape[0]
        ax1.yaxis.set_major_formatter(PercentFormatter(xmax=total_particles))
        ax1.set_xlim([0, 0.025])
        ax1.set_ylim([0, total_particles])
        ax1.set_xticks(bin_edges)
        ax1.set_yticks(np.linspace(0, total_particles, 10+1))

        # Add frequency on top of each bar
        for count, x_pos in zip(counts, bins[:-1]):
            ax1.text(
                x_pos + (bins[1] - bins[0]) / 2,
                count,
                f'{count:.0f}',
                ha='center',
                va='bottom'
            )


        # Add total weight plot
        ax2 = ax1.twinx()
        bin_weights = np.histogram(weights_particles, bins=bin_edges, weights=weights_particles)[0]
        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        ax2.plot(bin_centers, bin_weights, color='red', marker='o', label='Total Weight')

        ax2.set_ylabel('Total Weight')
        ax2.set_ylim([0, 1]) 
        ax2.set_yticks(np.linspace(0, 1, 10+1))

        ax1.grid(True)
        ax1.set_title(
            f'Particle Weights Histogram (ESS={Utils.compute_ess(weights_particles):.2f})')
        fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)


        file_path = os.path.join(os.path.abspath(os.getcwd()), 'outputs', file_name)
        plt.suptitle(file_name)
        plt.tight_layout()
        if save: plt.savefig(file_path, dpi=300)
        if show: plt.show()



class Simulation:
    def __init__(
            self,
            simulation_duration,
            dt_measurement,
            dt_prediction,
            n_particles,
            landmarks,
            x_estimation,
            x_odometry,
            x_true,
            P_estimation,
            Q_true,
            R_true,
        ):
        self.simulation_duration = simulation_duration
        self.n_particles = n_particles
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

        error = x_estimation - x_true
        error[2, 0] = Utils.convert_angle(error[2, 0])

        self.history_x_error = error
        self.history_x_covariance = P_estimation
        self.history_time = [0]


    def get_observation(
            self, k: int, black_out: bool = False
        ) -> list[np.ndarray, int] | list[None, None]:
        """
        Return a noisy observation of a random landmark at instant k.

        Args:
            k (int) : interation step.
            black_out (bool) : has data black-out? Default value is False.
        """
        np.random.seed(seed*3 + k)  # Ensuring random repexility for k

        if k * self.dt_prediction % self.dt_measurement == 0:
            if black_out:
                valid_measurement = False if 250 < k < 350 else True
            else:
                valid_measurement = True

            if not valid_measurement:
                return None, None
            else:
                landmark_index = np.random.randint(0, self.landmarks.shape[1] - 1)
                z_noise = np.sqrt(self.R_true) @ np.random.randn(2)
                z_noise = np.array([z_noise]).T

                z = PF.observation_model_prediction(self.x_true, landmark_index, self.landmarks)
                z += z_noise
                z[1, 0] = Utils.convert_angle(z[1, 0])

                return z, landmark_index

        return None, None


    def get_odometry(self, k: int) -> list[np.ndarray]:
        """
        Return a noisy odometry and command measurements at instant k as np.ndarrays.

        Args:
            k (int) : interation step.
        """
        np.random.seed(seed*2 + k)  # Ensuring random repexility for k

        u_noise = np.sqrt(self.Q_true) @ np.random.randn(3)
        u_noise = np.array([u_noise]).T
        u = self.get_robot_control(k) + self.dt_prediction * u_noise

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


    def plot(
            self,
            file_name: str,
            landmark_index: int,
            x_estimation: np.ndarray[float],
            x_particles: np.ndarray[float],
            weights_particles: np.ndarray[float],
            save: bool = False,
            show: bool = True,
        ) -> None:
        """"
        Plot simulation results.

        Args:
            file_name (str) : output image name.
            landmark_index (int) : reference landmark index.
            x_estimation (np.ndarray[float]) : system state estimation at instant k.
            x_particles (np.ndarray[float]) : particles coordinates at instant k.
            weights_particles (np.ndarray[float]) : weights of particles at instant k.
            save (bool) : save result? Default value is True.
            show (bool) : show result? Default value is True.
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
            label="Trajectory")
        if landmark_index != None:
            ax1.plot(
                [self.x_true[0][0], self.landmarks[0, landmark_index]],
                [self.x_true[1][0], self.landmarks[1, landmark_index]],
                "gray"
            )

        # Estimated odometry trajectory
        ax1.plot(
            self.history_x_odometry[0, :],
            self.history_x_odometry[1, :],
            "-g",
            label="Odometry"
        )

        # Estimated Particle Filter trajectory and current particles
        ax1.plot(
            self.history_x_estimation[0, :],
            self.history_x_estimation[1, :],
            "-y",
            label="Particle Filter"
        )
        ax1.plot(x_estimation[0], x_estimation[1], ".r")

        label_ess = f'ESS {Utils.compute_ess(weights_particles):2.4f}'
        ax1.scatter(x_particles[0, :], x_particles[1, :], s=weights_particles*10, label=label_ess)
        for i in range(self.n_particles):
            ax1.arrow(
                x_particles[0, i],
                x_particles[1, i],
                5 * np.cos(x_particles[2, i] + np.pi / 2),
                5 * np.sin(x_particles[2, i] + np.pi / 2),
                color = 'orange'
            )

        ax1.grid(True)
        axis_max = 60
        ticks = [i for i in range(-axis_max, +axis_max+1, 10)]
        ax1.axis([-axis_max, +axis_max, -axis_max, +axis_max])
        ax1.set_title('Cartesian Coordinates')
        ax1.set_ylabel('y [m]')
        ax1.set_xlabel('x [m]')
        ax1.set_xticks(ticks)
        ax1.set_yticks(ticks)
        ax1.legend(loc='upper left')

        # add common error markers
        ax2.set_xticks([])

        # plot errors curves
        x_ticks = [i for i in range(0, self.simulation_duration+1, 100)]

        rms_x_err = Utils.compute_rms_error(self.history_x_error[0, :])
        rms_y_err = Utils.compute_rms_error(self.history_x_error[1, :])
        rms_theta_err = Utils.compute_rms_error(self.history_x_error[2, :])

        rms_x_cov = Utils.compute_rms_error(3*self.history_x_covariance[0, :])
        rms_y_cov = Utils.compute_rms_error(3*self.history_x_covariance[1, :])
        rms_theta_cov = Utils.compute_rms_error(3*self.history_x_covariance[2, :])

        label_x_err = f'{rms_x_err:2.4f} error'
        label_y_err = f'{rms_y_err:2.4f} error'
        label_theta_err = f'{rms_theta_err:2.4f} error'

        label_x_cov = f'{rms_x_cov:2.4f} 3$\sigma$ covariance'
        label_y_cov = f'{rms_y_cov:2.4f} 3$\sigma$ covariance'
        label_theta_cov = f'{rms_theta_cov:2.4f} 3$\sigma$ covariance'

        # plot errors curves
        ax3.plot(times, self.history_x_error[0, :], 'b', label=label_x_err)
        ax3.plot(times, +3.0 * self.history_x_covariance[0, :], 'r', label=label_x_cov)
        ax3.plot(times, -3.0 * self.history_x_covariance[0, :], 'r')
        ax3.fill_between(
            times,
            +3.0 * self.history_x_covariance[0, :], 
            -3.0 * self.history_x_covariance[0, :],
            color='gray',
            alpha=0.75
        )
        ax3.set_title('Particle Filter')
        ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax3.set_ylabel('x [m]')
        ax3.set_xlim(0, self.simulation_duration)
        ax3.set_xticks(x_ticks)
        ax3.set_xticklabels(['' for _ in x_ticks])
        ax3.legend(loc='upper right')
        ax3.grid(True)

        ax4.plot(times, self.history_x_error[1, :], 'b', label=label_y_err)
        ax4.plot(times, +3.0 * self.history_x_covariance[1, :], 'r', label=label_y_cov)
        ax4.plot(times, -3.0 * self.history_x_covariance[1, :], 'r')
        ax4.fill_between(
            times,
            +3.0 * self.history_x_covariance[1, :], 
            -3.0 * self.history_x_covariance[1, :],
            color='gray',
            alpha=0.75
        )
        ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax4.set_ylabel('y [m]')
        ax4.set_xlim(0, self.simulation_duration)
        ax4.set_xticks(x_ticks)
        ax4.set_xticklabels(['' for _ in x_ticks])
        ax4.legend(loc='upper right')
        ax4.grid(True)

        ax5.plot(times, self.history_x_error[2, :], 'b', label=label_theta_err)
        ax5.plot(times, +3.0 * self.history_x_covariance[2, :], 'r', label=label_theta_cov)
        ax5.plot(times, -3.0 * self.history_x_covariance[2, :], 'r')
        ax5.fill_between(
            times,
            +3.0 * self.history_x_covariance[2, :], 
            -3.0 * self.history_x_covariance[2, :],
            color='gray',
            alpha=0.75
        )
        ax5.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax5.set_ylabel(r"$\theta$ [rad]")
        ax5.set_xlabel('time [s]')
        ax5.set_xlim(0, self.simulation_duration)
        ax5.set_xticks(x_ticks)
        ax5.legend(loc='upper right')
        ax5.grid(True)

        file_path = os.path.join(os.path.abspath(os.getcwd()), 'outputs', file_name)

        plt.suptitle(file_name)
        plt.tight_layout()
        if save: plt.savefig(file_path, dpi=300)
        if show: plt.show()


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
            k (int) : simulation instant.
            x_estimation (np.ndarray[float]) : system state estimation at instant k.
            P_estimation (np.ndarray[float]) : estimation covariance at instant k.
        """
        self.history_x_true = np.hstack((self.history_x_true, self.x_true))
        self.history_x_odometry = np.hstack((self.history_x_odometry, self.x_odometry))
        self.history_x_estimation = np.hstack((self.history_x_estimation, x_estimation))

        error = x_estimation - self.x_true
        error[2, 0] = Utils.convert_angle(error[2, 0])

        self.history_x_error = np.hstack((self.history_x_error, error))
        self.history_x_covariance = np.hstack((
            self.history_x_covariance, np.sqrt(np.diag(P_estimation).reshape(3, 1))
        ))
        self.history_time.append(k*self.dt_prediction)



def execution(
        dt_measurement: float = 1,
        dt_prediction: float = 1,
        n_landmarks: int = 5,
        n_particles: int = 300,
        resample_method: str = 'low-variance',
        theta_eff: float = 0.1,
        Q_constant: float = 2,
        R_constant: float = 2,
        black_out: bool = False,
        save: bool = False,
        show: bool = True,
    ) -> None:
    """
    Execute an Particles Filter simulation.

    Args:
        dt_measurement (float) : measurement interval in seconds. Default values is 1.
        dt_prediction (float) : prediction interval in seconds. Default values is 1.
        n_landmarks (int) : number of landmarks simulated. Default values is 30.
        resample_method (str) : resample method. Default values is 'low-variance'.
        theta_eff (float) : resampling threshould between 0 and 1. Default value is 1.
        Q_constant (float) : process noise covariance matrix Q constant. Default values is 1.
        R_constant (float) : measure noise covariance matrix R constant. Default values is 1.
        black_out (bool) : no measures between 2500 s and 3000s? Default values is False.
        save (bool) : save result? Default value is False.
        show (bool) : show result? Default value is True.
    """
    # Define Particle Filter covariance errors robot movements
    Q_true = np.diag([0.02, 0.02, 1*pi/180]) ** 2
    R_true = np.diag([0.5, 1*pi/180]) ** 2

    Q_estimation = Q_constant * np.eye(3, 3) @ Q_true
    R_estimation = R_constant * np.eye(2, 2) @ R_true


    # Simulation initial conditions
    simulation_duration = 1000  # simulation durantion [s]
    landmarks = 120*np.random.rand(2, n_landmarks)-60

    x_true = np.array([[1, -50, 0]]).T
    x_odometry = x_true
    x_particles = x_true + np.diag([1, 1, 0.1]) @ np.random.randn(3, n_particles)

    weights_particles = np.ones((n_particles))/n_particles
    weights_particles = weights_particles / np.sum(weights_particles)

    x_estimation = np.average(x_particles, axis=1, weights=weights_particles)
    x_estimation = np.expand_dims(x_estimation, axis=1)
    P_estimation = np.sqrt(np.average(
        (x_particles-x_estimation)*(x_particles-x_estimation), axis=1, weights=weights_particles
    ))
    P_estimation = np.expand_dims(P_estimation, axis=1)

    # Simulation environment
    simulation = Simulation(
        simulation_duration,
        dt_measurement,
        dt_prediction,
        n_particles,
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
        z, landmark_index = simulation.get_observation(k, black_out)

        # Particle Filter Prediction
        for p in range(simulation.n_particles):
            x_particles[:, p:p+1] = PF.motion_model_prediction(
                x_particles[:, p:p+1], u_tilde, simulation.dt_prediction, Q_estimation
            )

        # Particle Filter Correction
        if z is not None:
            for p in range(n_particles):
                h = PF.observation_model_prediction(
                    x_particles[:, p:p+1], landmark_index, simulation.landmarks
                )

                innovation = z - h
                innovation[1] = Utils.convert_angle(innovation[1])

                exponent = -0.5 * innovation.T @ np.linalg.inv(R_estimation) @ innovation
                normalizer = np.sqrt(np.linalg.det(2 * pi * R_estimation))

                # Compute particle weight using gaussian model
                weights_particles[p] = np.exp(exponent) / normalizer

        weights_particles /= np.sum(weights_particles)

        # Particle Filter Update
        x_estimation = np.mean(x_particles, axis=1)
        x_estimation = x_estimation.reshape(3,1)

        P_estimation = PF.P(x_estimation, x_particles)

        # Particle Filter Resampling
        if Utils.compute_ess(weights_particles) < n_particles * theta_eff:
            x_particles, weights_particles = PF.resample_particles(
                x_particles, weights_particles, simulation.n_particles, resample_method
            )

        # Update data history
        simulation.update_history(k, x_estimation, P_estimation)


    # Plot Simulation Results
    file_name = f'PF_{dt_measurement}_{dt_prediction}_'
    file_name += f'{n_landmarks:03d}_{n_particles:03d}_'
    file_name += f'{resample_method.lower()}_{theta_eff}_'
    file_name += f'{Q_constant}_{R_constant}_'
    file_name += f'{black_out}'
    file_name = file_name.replace('.', '-') + '.png'

    simulation.plot(
        file_name, landmark_index, x_estimation, x_particles, weights_particles, save, show
    )

    # Plot Weights Histogram
    file_name = f'hist_{dt_measurement}_{dt_prediction}_'
    file_name += f'{n_landmarks:03d}_{n_particles:03d}_'
    file_name += f'{resample_method.lower()}_{theta_eff}_'
    file_name += f'{Q_constant}_{R_constant}_'
    file_name += f'{black_out}'
    file_name = file_name.replace('.', '-') + '.png'

    Utils.weights_histogram(file_name, weights_particles, theta_eff, save, show)



def main():
    arr = [1]

    for var in arr:
        execution(resample_method='multinomial', save=True, show=False)



if __name__ == "__main__":
    main()
