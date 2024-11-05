"""
Extended Kalman Filter SLAM example

author: Atsushi Sakai (@Atsushi_twi)

Modified : Goran Frehse, David Filliat
"""

import math

import matplotlib.pyplot as plt
import numpy as np

DT = 0.1  # time tick [s]
SIM_TIME = 80.0  # simulation time [s]
MAX_RANGE = 10.0  # maximum observation range
M_DIST_TH = 9.0  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM state size [x,y]
KNOWN_DATA_ASSOCIATION = 1  # Whether we use the true landmarks id or not

# Simulation parameter
# noise on control input
Q_sim = (3 * np.diag([0.1, np.deg2rad(1)])) ** 2
# noise on measurement
Py_sim = (1 * np.diag([0.1, np.deg2rad(5)])) ** 2

# Kalman filter Parameters
# Estimated input noise for Kalman Filter
Q = 2 * Q_sim
# Estimated measurement noise for Kalman Filter
Py = 2 * Py_sim

# Initial estimate of pose covariance
initPEst = 0.01 * np.eye(STATE_SIZE)
initPEst[2,2] = 0.0001  # low orientation error

# True Landmark id for known data association
trueLandmarkId =[]

# Init displays
show_animation = True
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 7))
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


def plot_covariance_ellipse(xEst, PEst, axes, lineType):
    """
    Plot one covariance ellipse from covariance matrix
    """

    Pxy = PEst[0:2, 0:2]
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
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
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
                  [0.0, 1.0, float(DT * u[0,0] * math.cos(x[2, 0]))],
                  [0.0, 0.0, 1.0]])

    # Jacobian of f(X,u) wrt u
    B = np.array([[float(DT * math.cos(x[2, 0])), 0.0],
                  [float(DT * math.sin(x[2, 0])), 0.0],
                  [0.0, DT]])

    return A, B


# --- Observation model related functions

def observation(xTrue, xd, uTrue, Landmarks):
    """
    Generate noisy control and observation and update true position and dead reckoning
    """
    xTrue = motion_model(xTrue, uTrue)

    # add noise to gps x-y
    y = np.zeros((0, 3))

    for i in range(len(Landmarks[:, 0])):

        dx = Landmarks[i, 0] - xTrue[0, 0]
        dy = Landmarks[i, 1] - xTrue[1, 0]
        d = math.hypot(dx, dy)
        angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
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
    
    return xTrue, y, xd, u


def search_correspond_landmark_id(xEst, PEst, yi):
    """
    Landmark association with Mahalanobis distance
    """

    nLM = calc_n_lm(xEst)

    min_dist = []

    for i in range(nLM):
        innov, S, H = calc_innovation(xEst, PEst, yi, i)
        min_dist.append(innov.T @ np.linalg.inv(S) @ innov)
        

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
                   [math.sin(x[2,0] + y[1]), y[0] * math.cos(x[2,0] + y[1])]])

    return Jr, Jy


# --- Kalman filter related functions

def calc_innovation(xEst, PEst, y, LMid):
    """
    Compute innovation and Kalman gain elements
    """

    # Compute predicted observation from state
    lm = get_landmark_position_from_state(xEst, LMid)
    delta = lm - xEst[0:2]
    q = (delta.T @ delta)[0, 0]
    y_angle = math.atan2(delta[1, 0], delta[0, 0]) - xEst[2, 0]
    yp = np.array([[math.sqrt(q), pi_2_pi(y_angle)]])

    # compute innovation, i.e. diff with real observation
    innov = (y - yp).T
    innov[1] = pi_2_pi(innov[1])

    # compute matrixes for Kalman Gain
    H = jacob_h(q, delta, xEst, LMid)
    S = H @ PEst @ H.T + Py
    
    return innov, S, H


def ekf_slam(xEst, PEst, u, y):
    """
    Apply one step of EKF predict/correct cycle
    """
    
    S = STATE_SIZE
    
    # Predict
    A, B = jacob_motion(xEst[0:S], u)

    xEst[0:S] = motion_model(xEst[0:S], u)

    PEst[0:S, 0:S] = A @ PEst[0:S, 0:S] @ A.T + B @ Q @ B.T
    PEst[0:S,S:] = A @ PEst[0:S,S:]
    PEst[S:,0:S] = PEst[0:S,S:].T

    PEst = (PEst + PEst.T) / 2.0  # ensure symetry
    
    # Update
    for iy in range(len(y[:, 0])):  # for each observation
        nLM = calc_n_lm(xEst)
        
        if KNOWN_DATA_ASSOCIATION:
            try:
                min_id = trueLandmarkId.index(y[iy, 2])
            except ValueError:
                min_id = nLM
                trueLandmarkId.append(y[iy, 2])
        else:
            min_id = search_correspond_landmark_id(xEst, PEst, y[iy, 0:2])


        # Extend map if required
        if min_id == nLM:
            print("New LM")
            
            # Extend state and covariance matrix
            xEst = np.vstack((xEst, calc_landmark_position(xEst, y[iy, :])))

            Jr, Jy = jacob_augment(xEst[0:3], y[iy, :])
            bottomPart = np.hstack((Jr @ PEst[0:3, 0:3], Jr @ PEst[0:3, 3:]))
            rightPart = bottomPart.T
            PEst = np.vstack((np.hstack((PEst, rightPart)),
                              np.hstack((bottomPart,
                              Jr @ PEst[0:3, 0:3] @ Jr.T + Jy @ Py @ Jy.T))))

        else:
            # Perform Kalman update
            innov, S, H = calc_innovation(xEst, PEst, y[iy, 0:2], min_id)
            K = (PEst @ H.T) @ np.linalg.inv(S)
            
            xEst = xEst + (K @ innov)
                        
            PEst = (np.eye(len(xEst)) - K @ H) @ PEst
            PEst = 0.5 * (PEst + PEst.T)  # Ensure symetry
        
    xEst[2] = pi_2_pi(xEst[2])

    return xEst, PEst


# --- Main script

def main():
    print(__file__ + " start!!")

    time = 0.0

    # Define landmark positions [x, y]
    Landmarks = np.array([[0.0, 5.0],
                          [11.0, 1.0],
                          [3.0, 15.0],
                          [-5.0, 20.0]])

    # Init state vector [x y yaw]' and covariance for Kalman
    xEst = np.zeros((STATE_SIZE, 1))
    PEst = initPEst

    # Init true state for simulator
    xTrue = np.zeros((STATE_SIZE, 1))

    # Init dead reckoning (sum of individual controls)
    xDR = np.zeros((STATE_SIZE, 1))

    # Init history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hxError = np.abs(xEst-xTrue)  # pose error
    hxVar = np.sqrt(np.diag(PEst[0:STATE_SIZE,0:STATE_SIZE]).reshape(3,1))  #state std dev


    # counter for plotting
    count = 0

    while  time <= SIM_TIME:
        count = count + 1
        time += DT

        # Simulate motion and generate u and y
        uTrue = calc_input()
        xTrue, y, xDR, u = observation(xTrue, xDR, uTrue, Landmarks)

        xEst, PEst = ekf_slam(xEst, PEst, u, y)

        # store data history
        hxEst = np.hstack((hxEst, xEst[0:STATE_SIZE]))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        err = xEst[0:STATE_SIZE]-xTrue
        err[2] = pi_2_pi(err[2])
        hxError = np.hstack((hxError,err))
        hxVar = np.hstack((hxVar,np.sqrt(np.diag(PEst[0:STATE_SIZE,0:STATE_SIZE]).reshape(3,1))))


        if show_animation and count%15==0:
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            
            ax1.cla()
            
            # Plot true landmark and trajectory
            ax1.plot(Landmarks[:, 0], Landmarks[:, 1], "*k")
            ax1.plot(hxTrue[0, :], hxTrue[1, :], "-k", label="True")

            # Plot odometry trajectory
            ax1.plot(hxDR[0, :], hxDR[1, :], "-g", label="Odom")

            # Plot estimated trajectory, pose and landmarks
            ax1.plot(hxEst[0, :], hxEst[1, :], "-r", label="EKF")
            ax1.plot(xEst[0], xEst[1], ".r")
            plot_covariance_ellipse(xEst[0: STATE_SIZE],
                                    PEst[0: STATE_SIZE, 0: STATE_SIZE], ax1, "--r")

            for i in range(calc_n_lm(xEst)):
                id = STATE_SIZE + i * 2
                ax1.plot(xEst[id], xEst[id + 1], "xr")
                plot_covariance_ellipse(xEst[id:id + 2],
                                        PEst[id:id + 2, id:id + 2], ax1, "--r")



            ax1.axis([-12, 12, -2, 22])
            ax1.grid(True)
            ax1.legend()
            
            # plot errors curves
            ax3.plot(hxError[0, :],'b')
            ax3.plot(3.0 * hxVar[0, :],'r')
            ax3.plot(-3.0 * hxVar[0, :],'r')
            ax3.set_ylabel('x')
            ax3.set_title('Real error (blue) and 3 $\sigma$ covariances (red)')
            
            ax4.plot(hxError[1, :],'b')
            ax4.plot(3.0 * hxVar[1, :],'r')
            ax4.plot(-3.0 * hxVar[1, :],'r')
            ax4.set_ylabel('y')

            ax5.plot(hxError[2, :],'b')
            ax5.plot(3.0 * hxVar[2, :],'r')
            ax5.plot(-3.0 * hxVar[2, :],'r')
            ax5.set_ylabel(r"$\theta$")

            plt.pause(0.001)


    plt.savefig('EKFSLAM.png')

    tErrors = np.sqrt(np.square(hxError[0, :]) + np.square(hxError[1, :]))
    oErrors = np.sqrt(np.square(hxError[2, :]))
    print("Mean (var) translation error : {:e} ({:e})".format(np.mean(tErrors), np.var(tErrors)))
    print("Mean (var) rotation error : {:e} ({:e})".format(np.mean(oErrors), np.var(oErrors)))    # keep window open
    print("Press Q in figure to finish...")
    plt.show()

if __name__ == '__main__':
    main()
