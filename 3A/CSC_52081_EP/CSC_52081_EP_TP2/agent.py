# if you are using Google Colab, uncomment the following line (comment again when you have run it)
# !wget https://www.lix.polytechnique.fr/~jread/courses/inf581/labs/01/environment.py

from environment import Environment

import numpy as np
import matplotlib.pyplot as plt

def generate_trajectory(env, T:int = 5) -> tuple[np.ndarray[float], np.ndarray[float]]:
    '''
    Generate a random path of length T with associated observations.

    Parameters
    ----------

        T (int): path length

    Returns
    -------

        o (np.ndarray): Tx2 array of observations
        s (np.ndarray): Tx1 array of states
    '''
    s = np.zeros(T, dtype=int)
    o = np.zeros((T, 2), dtype=int)

    s[0], o[0] = env.step()

    for t in range(1, T):
        s_t, o_t = env.step(s[t-1])

        s[t] = s_t
        o[t] = o_t

    return o, s

class Agent:
    def __init__(self, env: Environment) -> None:
        '''
        Parameters
        ----------
            env (Environment): environment provided
        '''
        self.env = env


    def P_trajectories(self, observations: np.ndarray[float], M: int = -1) -> dict:
        '''
        Provides full conditional distribution P(states | observations) where states and
        observations are sequences of length T.

        Parameters
        ----------
            observations (np.ndarray[float]): Tx2 sequence of observations of dimension.

            M (int, optional) : computation method:

                -1: use Brute Force solution
                >0: use M Monte Carlo simulations 

        Returns
        -------
            p (dict[str: float]) : probability distribution p[states] = P(states | observations).
        '''
        p: dict = {}
        rustle: int = 1
        crinkle: int = 0

        # Exact Recovery with brute force
        if M == -1:
            def generate_paths(current_path, current_probability, step):
                """Recursive helper function to generate all paths."""
                if step == len(observations):
                    path: str = " ".join(map(str, current_path))
                    p[path] = current_probability

                    return

                current_state: int = current_path[-1]

                for next_state in range(self.env.n_states):
                    transition_probability: float = self.env.P_S[current_state, next_state]

                    if transition_probability != 0:
                        noise_probability: float  = (
                            self.env.P_O[next_state, rustle, observations[step][rustle]]
                            * self.env.P_O[next_state, crinkle, observations[step][crinkle]]
                        )

                        generate_paths(
                            current_path + [next_state],
                            current_probability * transition_probability * noise_probability,
                            step + 1,
                        )


            for initial_state in range(self.env.n_states):
                initial_probability: float = self.env.P_1[initial_state]

                if initial_probability != 0:
                    generate_paths([initial_state], initial_probability, 1)

            total_probability: float = sum(p.values())
            if total_probability > 0:
                p = {k: v / total_probability for k, v in p.items() if v > 0.0}


        # Monte Carlo simulation
        elif M > 0:
            for _ in range(M):
                initial_state: int = np.random.choice(self.env.n_states, p=self.env.P_1)

                current_path: list = [initial_state]
                current_probability: float = self.env.P_1[initial_state]

                for step in range(len(observations)):
                    current_state: int = current_path[-1]

                    next_state: int = np.random.choice(
                        self.env.n_states,
                        p = self.env.P_S[current_state]
                    )

                    transition_probability: float = self.env.P_S[current_state, next_state]

                    noise_probability: float = (
                        self.env.P_O[next_state, rustle, observations[step][rustle]] *
                        self.env.P_O[next_state, crinkle, observations[step][crinkle]]
                    )

                    current_path.append(next_state)
                    current_probability *= transition_probability * noise_probability

                path: str = " ".join(map(str, current_path))
                if path in p:
                    p[path] += current_probability
                else:
                    p[path] = current_probability

            total_probability: float = sum(p.values())
            if total_probability > 0:
                p = {k: v / total_probability for k, v in p.items() if v > 0.0}

        return p


    def P_states(
            self, observations: np.ndarray[float], instant: int = -1, M: int = -1
        ) -> list[float]:
        '''
        Provide P(state_t | observations) given observations o from 1,...,T.

        Parameters
        ----------

            observations (np.ndarray[float]): Tx2 sequence of observations of dimension.
            instant (int, optional): instant of observations. Default is '-1', last instant.
            M (int, optional) : computation method. Default is '-1'.
                -1: use Brute Force solution
                >0: use M Monte Carlo simulations 

        Returns
        -------

            P_states (list[float]) : probability distribution p[states] = P(states | observations).
        '''
        P_states: dict = {state: 0 for state in range(self.env.n_states)}

        if instant <= -1 or instant >= len(observations):
            instant = len(observations) - 1

        next_instant:int = instant + 1


        P_trajectories: dict = self.P_trajectories(observations[:next_instant, :], M)

        for trajectory, probability in P_trajectories.items():
            P_states[int(trajectory.split(" ")[instant])] += probability

        total_probability: float = sum(P_states.values())
        if total_probability > 0:
            P_states = {k: v / total_probability for k, v in P_states.items()}

        return np.array(list(P_states.values()))


    def Q_values(
            self, observations: np.ndarray[float], instant: int = -1, M: int = -1
        ) -> np.ndarray[float]:
        '''
        Provide expected reward for an action given an array of observations.

        Parameters
        ----------

            observations (np.ndarray[float]): Tx2 sequence of observations of dimension.
            instant (int, optional): instant of observations. Default is '-1', last instant.
            M (int, optional) : computation method. Default is '-1'.
                -1: use Brute Force solution
                >0: use M Monte Carlo simulations

        Returns
        -------

            Q_states (np.ndarray[float]) : expected reward for an action given observations.
        '''
        Q_values: np.ndarray = np.zeros(self.env.n_states)

        P_states = self.P_states(observations, instant, M)

        maximum_probability = np.max(P_states)
        for i, state_probability in enumerate(P_states):
            if state_probability == maximum_probability:
                Q_values[i] = 1

        return Q_values

    def action(
            self, observations: np.ndarray[float], instant: int = -1, M: int = -1
        ):
        '''
        Decide on the best action to take, under the provided observation. 

        Parameters
        ----------

            observations (np.ndarray[float]): Tx2 sequence of observations of dimension.
            instant (int, optional): instant of observations. Default is '-1', last instant.
            M (int, optional) : computation method. Default is '-1'.
                -1: use Brute Force solution
                >0: use M Monte Carlo simulations

        Returns
        -------

            bast action to take.
        '''
        Q_values = self.Q_values(observations, instant, M)

        possible_actions = []

        maximum_probability = np.max(Q_values)
        for i, state_probability in enumerate(Q_values):
            if state_probability == maximum_probability:
                possible_actions.append(i)

        return np.random.choice(possible_actions)


