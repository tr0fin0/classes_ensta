from matplotlib.colors import ListedColormap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



class Environment():
    def __init__(
            self, grid: np.ndarray[float], noises: list[float] = [0.9, 0.8], fps: bool = False
        ):
        '''
        Environment initialization.

        Args:
            grid (np.ndarray[float]): n x m matrix defining environment
            noises (list[float]): noise occurrence probability vector. Default is '[0.9, 0.8]'
            fps (bool): environment allows false positives? Default is 'False'.
        '''
        # Grid Initialization
        self.grid = grid

        self.n_rows = self.grid.shape[0]
        self.n_columns = self.grid.shape[1]
        self.n_states = int(self.n_rows * self.n_columns)

        # State Space Initialization
        self.labels = np.arange(1,self.n_states+1)

        self.noises = noises
        self.n_noises = len(self.noises)

        self.false_positives = fps

        # State Transition Matrix, probability from moving between states
        self.P_S = np.zeros((self.n_states, self.n_states))

        # Prior Vector, initial state probability
        self.P_1 = np.zeros(self.n_states)

        # Observation Matrix, conditional probability of observing a feature
        self.P_O = np.zeros((self.n_states, self.n_noises, 2))


        arr_noises = np.vstack((1 - np.array(self.noises), np.array(self.noises))).T

        for state in range(self.n_states):
            i, j = divmod(state, self.n_columns)

            # entry point
            if self.grid[i, j] > 3:
                self.P_1[state] = 1

            if self.false_positives:
                # 10% chance of false positive
                self.P_O[state, 0] = [0.9, 0.1]
                self.P_O[state, 1] = [0.9, 0.1]
            else:
                self.P_O[state, 0] = [1, 0]
                self.P_O[state, 1] = [1, 0]

            # sound rustle
            if self.grid[i, j] == 2 or self.grid[i, j] == 3:
                self.P_O[state, 0] = arr_noises[0]

            # sound crinkle
            if self.grid[i, j] == 1 or self.grid[i, j] == 3:
                self.P_O[state, 1] = arr_noises[1]

            # state transition
            if i > 0:
                self.P_S[state, state - self.n_columns] = 1

            if i < self.n_rows - 1:
                self.P_S[state, state + self.n_columns] = 1

            if j > 0:
                self.P_S[state, state - 1] = 1

            if j < self.n_columns - 1 and state < self.n_states - 1:
                self.P_S[state, state + 1] = 1
            self.P_S[state, :] = self.P_S[state, :] / sum(self.P_S[state, :])

            # probability normalization
            assert(sum(self.P_S[state]) == 1)
            assert(sum(self.P_O[state, 0]) == 1)
            assert(sum(self.P_O[state, 1]) == 1)

        self.P_1 = self.P_1 / sum(self.P_1)


    def __str__(self) -> str:
        '''
        Return a string representation of class environment.
        '''
        description = f"{self.__class__.__name__}:\n"
        description += f"\tFPS:\t[{int(self.false_positives):4d}]\n"
        description += f"\tnoises:\t[{self.n_noises:4d}] {self.noises}\n"
        description += f"\tlabels:\t[{self.n_states:4d}] \n"
        description += f"\tP_S:\n"


        rows = []
        for row in self.P_S:
            rows.append(" ".join(f"{value:.2f}" for value in row))

        description += "\n".join(rows)

        return description


    def rwd(self, s, a):
        '''
        Reward function r(s, a) of taking action a when in state s

        Parameters
        ----------
        s : int
            true state (tile which contains the object)
        a : int
            estimated state

        Returns
        -------
        float
            reward obtained from taking action a given state s
    '''
        return (s==a)*1.

    def step(self, previous_state: int | None = None) -> tuple[int, np.ndarray[int]]:
        '''
        Return next state s and observation o from previous state previous_state.

        Args:
            previous_state (int | None): previous state. Default is 'None'.

        Returns:
            s (int) : next state.
            o (np.ndarray[int]) : state observation
        '''
        # Generate a State: s' ~ p( . | s)
        if previous_state is None:
            next_state = np.random.choice(self.n_states, p=self.P_1)
        else:
            next_state = np.random.choice(self.n_states, p=self.P_S[previous_state,:])

        # Generate an Observation: o' ~ p(. | s')
        observation = np.zeros(self.n_noises)
        for j in range(self.n_noises):
            w = self.P_O[next_state, j]
            observation[j] = np.random.choice(self.n_noises, p = w)

            #print("P(O | s=%s) = %s" % (str(s),str(self.P_O[s,j,:])))

        return next_state, observation


    def tile2cell(self, s):
        return divmod(s, self.n_columns)

    def render(
        self,
        y_seq=None,
        x_seq=None,
        dgrid: np.ndarray[float] = None,
        a_star=None,
        paths=[],
        title=None,
        add_legend=True,
        output_fname=None
    ):
        '''
        Plot a visual representation of the environment.

        Parameters
        ----------

        y_seq : numpy array (dtype=int)
            a path (e.grid., [1,3,1,2])

        x_seq :
            observations associated with the path

        dgrid : shape like self.grid
            contains values (e.grid., probabilities) to show in each tile

        a_star : int
            the optimal action

        title : str
            a title for the plot
        '''

        fig, ax = plt.subplots(figsize=[8,4])

        colors = {
            0 : "white",    # nothing
            1 : "green",    # sound 1
            2 : "red",      # sound 2
            3 : "orange",   # sound 1 + 2
            4 : "yellow"    # entry
        }
        labels = ['', 'Crinkle', 'Rustle', 'Crinkle/rustle', 'Entry']

        # Plot the tiles in the room ...

        if dgrid is None:
            # ... as a visual representation
            im = ax.imshow(self.grid, cmap=ListedColormap(list(colors.values())), alpha=0.3)
            patches = [mpatches.Patch(color=colors[i], alpha=0.3, label=labels[i]) for i in [1,2,3,4]]
            if add_legend:
                plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        else:
            # ... as a probability mass function
            im = ax.imshow(dgrid.reshape(self.n_rows, self.n_columns), cmap=plt.cm.Reds)

        # Plot the path, alongside the observations generated by that path, and the optimal action.

        if y_seq is not None:

            # Draw the path
            T = len(y_seq)
            y_coords = np.array([self.tile2cell(y_t)[0] for y_t in y_seq]) + np.random.randn(T)*0.1
            x_coords = np.array([self.tile2cell(y_t)[1] for y_t in y_seq]) + np.random.randn(T)*0.1
            ax.plot(x_coords,y_coords,"ko-")
            ax.plot(x_coords[-1],y_coords[-1],"kx",markersize=20)

            # Draw the action (i.e., target tile)
            if a_star is not None:
                y_coord = self.tile2cell(a_star)[0]
                x_coord = self.tile2cell(a_star)[1]
                ax.plot(x_coord,y_coord,"m+",markersize=15)

            # Draw the sounds (observations)
            if x_seq is not None:
                ax.scatter(np.array(x_coords)[x_seq[:, 0] > 0], np.array(y_coords)[x_seq[:, 0] > 0], marker='o', s=200, facecolors='none', linewidths=3, edgecolors=colors[2])
                ax.scatter(np.array(x_coords)[x_seq[:, 1] > 0], np.array(y_coords)[x_seq[:, 1] > 0], marker='o', s=400, facecolors='none', linewidths=3, edgecolors=colors[1])


        for path in paths:
            # Draw the path
            T = len(path)
            y_coords = np.array([self.tile2cell(s)[0] for s in path]) + np.random.randn(T)*0.1
            x_coords = np.array([self.tile2cell(s)[1] for s in path]) + np.random.randn(T)*0.1
            ax.plot(x_coords,y_coords,"mo:")
            ax.plot(x_coords[-1],y_coords[-1],"mx",markersize=10)


        # Ticks and grid

        ax.set_xticks(np.arange(0, self.n_columns, 1))
        ax.set_xticks(np.arange(-0.5, self.n_columns, 1), minor=True)
        ax.set_xticklabels(np.arange(0, self.n_columns, 1))

        ax.set_yticks(np.arange(0, self.n_rows, 1))
        ax.set_yticks(np.arange(-0.5, self.n_rows, 1), minor=True)
        ax.set_yticklabels(np.arange(0, self.n_rows, 1))

        ax.grid(which='minor', color='k')

        n = 0
        for i in range(self.n_rows):
            for j in range(self.n_columns):
              ax.text(j, i, n, va='center', ha='center')
              n = n + 1


        # Title

        if title is not None:
            ax.set_title(title)


        # Return

        plt.tight_layout()

        if output_fname is not None:
            plt.savefig(output_fname)

        return fig, ax


if __name__ == "__main__":

    G = np.array([[1,3,0,2,4,1],
                  [2,1,0,3,0,3],
                  [4,0,3,0,2,0],
                  [3,1,2,3,0,4],
                  [2,0,0,0,1,1]])

    env = Environment(G)
    s, o = env.step()
    ooo = np.array([o])
    sss = np.array([s]).reshape(1,-1)
    fig, ax = env.render(sss, ooo)
    #fig, ax = env.render(output_fname="environment.png")
    plt.show()
