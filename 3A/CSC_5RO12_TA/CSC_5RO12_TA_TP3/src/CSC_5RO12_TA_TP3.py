""""CSC_5RO12_TA_TP3, Navegation pour la Robotique algorithm."""

from dataclasses import dataclass
from math import sin, cos, atan2, pi
from matplotlib.ticker import FormatStrFormatter

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
            x_STD,
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

        # Init history matrixes
        self.history_x_estimation = x_estimation
        self.history_x_odometry = x_odometry
        self.history_x_true = x_true

        error = x_estimation - x_true
        error[2, 0] = Utils.wrap_angle(error[2, 0])

        self.history_x_error = error
        self.history_x_covariance = x_STD
        self.history_time = [0]

    # return true control at step k
    def get_robot_control(self, k):
        # generate  sin trajectory
        u = np.array([[0, 0.025,  0.1*np.pi / 180 * sin(3*np.pi * k / self.n_steps)]]).T
        return u


    # simulate new true robot position
    def simulate_world(self, k):
        dt_prediction = self.dt_prediction
        u = self.get_robot_control(k)
        self.x_true = Utils.compute_motion(self.x_true, u, dt_prediction)
        self.x_true[2, 0] = Utils.wrap_angle(self.x_true[2, 0])


    # computes and returns noisy odometry
    def get_odometry(self, k):
        # Ensuring random repetability for given k
        np.random.seed(seed*2 + k)

        # Model
        dt_prediction = self.dt_prediction
        u = self.get_robot_control(k)
        xnow = Utils.compute_motion(self.x_odometry, u, dt_prediction)
        uNoise = np.sqrt(self.Q_true) @ np.random.randn(3)
        uNoise = np.array([uNoise]).T
        xnow = Utils.compute_motion(xnow, uNoise, dt_prediction)
        self.x_odometry = xnow
        u_tilde = u + dt_prediction*uNoise
        return xnow, u_tilde


    # generate a noisy observation of a random feature
    def get_observation(self, k):
        # Ensuring random repetability for given k
        np.random.seed(seed*3 + k)

        # Model
        if k*self.dt_prediction % self.dt_measurement == 0:
            notValidCondition = False # False: measurement valid / True: measurement not valid
            if notValidCondition:
                z = None
                landmark_index = None
            else:
                landmark_index = np.random.randint(0, self.landmarks.shape[1] - 1)
                zNoise = np.sqrt(self.R_true) @ np.random.randn(2)
                zNoise = np.array([zNoise]).T
                z = PF.observation_model(self.x_true, landmark_index, self.landmarks) + zNoise
                z[1, 0] = Utils.wrap_angle(z[1, 0])
        else:
            z = None
            landmark_index = None
        return [z, landmark_index]
    

    def update_history(
            self, k: int, x_estimation: np.ndarray[float], x_STD: np.ndarray[float]
        ) -> None:
        """
        Update simulation history.

        Args:
            k (int) :simulation instant.
            x_estimation (np.ndarray[float]) : system state estimation at instant k.
            x_STD (np.ndarray[float]) : estimation covariance at instant k.
        """
        self.history_x_true = np.hstack((self.history_x_true, self.x_true))
        self.history_x_odometry = np.hstack((self.history_x_odometry, self.x_odometry))
        self.history_x_estimation = np.hstack((self.history_x_estimation, x_estimation))

        error = x_estimation - self.x_true
        error[2, 0] = Utils.wrap_angle(error[2, 0])

        self.history_x_error = np.hstack((self.history_x_error, error))
        self.history_x_covariance = np.hstack((self.history_x_covariance, x_STD))
        self.history_time.append(k*self.dt_prediction)


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
            x_estimation (np.ndarray[float]) : system state estimation at instant k.
            P_estimation (np.ndarray[float]) : estimation covariance at instant k.
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
                "-b"
            )

        # Estimated odometry trajectory
        ax1.plot(
            self.history_x_odometry[0, :],
            self.history_x_odometry[1, :],
            "-g",
            label="Odometry"
        )

        # Estimated EKF trajectory and current particles
        ax1.plot(
            self.history_x_estimation[0, :],
            self.history_x_estimation[1, :],
            "-r",
            label="Particle Filter"
        )
        ax1.plot(x_estimation[0], x_estimation[1], ".r")
        # TODO ? make utils function
        ax1.scatter(x_particles[0, :], x_particles[1, :], s=weights_particles*10)
        for i in range(self.n_particles):
            ax1.arrow(x_particles[0, i], x_particles[1, i], 5*np.cos(x_particles[2, i]+np.pi/2), 5*np.sin(x_particles[2, i]+np.pi/2), color = 'orange')

        ax1.grid(True)
        ax1.axis([-100, 100, -100, 100])
        ax1.set_title('Cartesian Coordinates')
        ax1.set_ylabel('y [m]')
        ax1.set_xlabel('x [m]')
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
        ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax4.grid(True)
        ax4.set_ylabel('y [m]')
        ax4.set_xlim(0, self.simulation_duration)
        ax4.set_xticks(x_ticks)
        ax4.set_xticklabels(['' for _ in x_ticks])
        ax4.legend(loc='upper right')
        ax4.grid(True)

        ax5.plot(times, self.history_x_error[2, :], 'b', label=label_theta_err)
        ax5.plot(times, +3.0 * self.history_x_covariance[2, :], 'r', label=label_theta_cov)
        ax5.plot(times, -3.0 * self.history_x_covariance[2, :], 'r')
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



@dataclass
class PF:
    """Particle Filter questions."""
    # evolution model (f)
    def motion_model(x: np.ndarray[float, float], u_tilde: np.ndarray[float, float], dt: int, Q_estimation: np.ndarray[float, float]) -> np.ndarray[float, float]:
        """
        Return motion model of agent.

        Note: based on CoursPF_Merlinge, slide 33.

        Args:
            x (np.ndarray[float, float]) : agent state at instant k from ground reference. (x, y, theta)
            u_tilde (np.ndarray[float, float]) : agent noisy odometry at instant k from robot reference. (vx, vy, omega)
            dt (int) : discrete time interval.
            Q_estimation (np.ndarray[float, float]) : .... estimation
        """
        x_k, y_k, theta_k = x[0, 0], x[1, 0], x[2, 0]
        vx_k, vy_k, omega_k = u_tilde[0, 0], u_tilde[1, 0], u_tilde[2, 0]

        w_vx_k, w_vy_k, w_omega_k = 0, 0, 0 # TODO define function from Q_estimation
        e_vx_k, e_vy_k, e_omega_k = vx_k + w_vx_k, vy_k + w_vy_k, omega_k + w_omega_k

        x_prediction = np.zeros_like(x)
        x_prediction[0, 0] = x_k + (e_vx_k * np.cos(theta_k) - e_vy_k * np.sin(theta_k)) * dt
        x_prediction[1, 0] = y_k + (e_vx_k * np.sin(theta_k) + e_vy_k * np.cos(theta_k)) * dt
        x_prediction[2, 0] = theta_k + e_omega_k * dt

        # chatGPT advice below
        # x_prediction = x + dt_prediction * u_tilde  # Simple linear model
        # x_prediction += np.sqrt(Q_estimation) @ np.random.randn(3)  # Add noise

        return x_prediction


    # observation model (h)
    def observation_model(xVeh, landmark_index, landmarks):
        # xVeh: vecule state
        # landmark_index: observed amer index
        # landmarks: map of all amers
        # slide 33

        # Landmark position
        landmark = landmarks[:, landmark_index]

        # Compute the expected observation (range and bearing to the landmark)
        dx = landmark[0] - xVeh[0]
        dy = landmark[1] - xVeh[1]
        expected_range = np.sqrt(dx**2 + dy**2)
        expected_bearing = atan2(dy, dx) - xVeh[2]

        # Return the expected observation
        return np.array([[expected_range], [expected_bearing]])


    # ---- particle filter implementation ----

    # Particle filter resampling
    def re_sampling(px, pw, n_particles):
        """
        low variance re-sampling
        """
        # slide 25

        w_cum = np.cumsum(pw)
        base = np.arange(0.0, 1.0, 1 / n_particles)
        re_sample_id = base + np.random.uniform(0, 1 / n_particles)
        indexes = []
        ind = 0
        for ip in range(n_particles):
            while re_sample_id[ip] > w_cum[ind]:
                ind += 1
            indexes.append(ind)

        px = px[:, indexes]
    #    pw = pw[indexes]

        # Normalization
        pw = np.ones(pw.shape)
        pw = pw / np.sum(pw)

        return px, pw


@dataclass
class Utils:
    # fit angle between -pi and pi
    def wrap_angle(a):
        if (a > np.pi):
            a = a - 2 * pi
        elif (a < -np.pi):
            a = a + 2 * pi
        return a


    # composes two transformations
    def compute_motion(tab, tbc, dt):
        assert tab.ndim == 2 # eg: robot state [x, y, heading]
        assert tbc.ndim == 2 # eg: robot control [Vx, Vy, angle rate]
        #dt : time-step (s)

        angle = tab[2, 0] + dt * tbc[2, 0] # angular integration by Euler

        angle = Utils.wrap_angle(angle)
        s = sin(tab[2, 0])
        c = cos(tab[2, 0])
        position = tab[0:2] + dt * np.array([[c, -s], [s, c]]) @ tbc[0:2] # position integration by Euler
        out = np.vstack((position, angle))

        return out


    def compute_rms_error(arr: np.ndarray[float]) -> float:
        """
        Return RMS error of an np.ndarray as a float.

        Args:
            arr (np.ndarray) : array to analyze.
        """
        return np.sqrt(np.mean(arr**2))



def execution(
        dt_measurement: int = 1,
        dt_prediction: int = 1,
        n_landmarks: int = 5,
        n_particles: int = 300,
        Q_constant: int = 2,
        R_constant: int = 2,
        save_result: bool = False,
        show_result: bool = True,
    ) -> None:
    """
    Execute an Particles Filter simulation.

    Args:
        dt_measurement (int) : measurement interval in seconds. Default values is 1.
        dt_prediction (int) : prediction interval in seconds. Default values is 1.
        n_landmarks (int) : number of landmarks simulated. Default values is 30.
        Q_constant (int) : process noise covariance matrix Q constant. Default values is 1.
        R_constant (int) : measure noise covariance matrix R constant. Default values is 1.
        save_result (bool) : save result? Default value is True.
        show_result (bool) : show result? Default value is True.
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
    x_STD = np.sqrt(np.average((x_particles-x_estimation)*(x_particles-x_estimation), axis=1, weights=weights_particles))
    x_STD = np.expand_dims(x_STD, axis=1)

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
        x_STD,
        Q_true,
        R_true,
    )


    # Loopo simulation environment
    for k in range(1, simulation.n_steps):
        simulation.simulate_world(k)    # Simulate robot motion
        x_odometry, u_tilde = simulation.get_odometry(k)

        # do prediction
        # for each particle we add control vector AND noise

        # slide 25
        # TODO
        x_tmp = PF.motion_model(x_odometry, u_tilde, dt_prediction, Q_estimation)

        # ...................

        # observe a random feature
        [z, landmark_index] = simulation.get_observation(k)

        if z is not None:
            for p in range(n_particles):
                # Predict observation from the particle position
        # slide 25
        # TODO
                z_prediction = 0

                # Innovation : perception error
        # slide 25
        # TODO
                innovation = np.array([0, 0, 0])
                innovation[1] = Utils.wrap_angle(innovation[1])

                # Compute particle weight using gaussian model
        # slide 25
        # TODO
                weights_particles[p] = 0
        # Normalization
        # TODO
        weights_particles = np.ones((n_particles))/n_particles


        # slide 25
        # Compute position as weighted mean of particles
        # TODO
        x_estimation = np.vstack([0, 0, 0])

        # slide 25
        # Compute particles std deviation
        # TODO
        P_estimation = 0 # Empirical covariance matrix
        x_STD = np.vstack([0, 0, 0]) # Column vector of standard deviations (sqrt of diagonal of P_estimation)


        # Reampling
        theta_eff = 0.1
        Nth = n_particles * theta_eff
        Neff = 0
        if Neff < Nth:
        # TODO
            pass
        # slide 25
            # Particle resampling
            # x_particles, weights_particles = np.array([0, 0, 0])


        # Update data history
        simulation.update_history(k, x_estimation, x_STD)

    file_name = f'PF_{dt_measurement}_{dt_prediction}_'
    file_name += f'{n_landmarks}_'
    file_name += f'{Q_constant}_{R_constant}.png'

    simulation.plot(file_name, landmark_index, x_estimation, x_particles, weights_particles, save_result, show_result)



def main():
    arr = [1]

    for var in arr:
        execution(show_result=True, save_result=False)



if __name__ == "__main__":
    main()
