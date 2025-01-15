import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

class Environment():

    def __init__(self, G, theta = [0.9,0.8], fps=False):
        '''
            Environment.

            Parameters
            ----------

            G : array_like(int, ndim=2) of shape (n_rows,n_columns)
                Specifies a grid where G[j,k] = entry & sound1 & sound2

            theta : array_like(float, ndim=1)
                Specifies the grid dynamics (acoustics)

        '''
        # Grid
        self.G = G

        # Grid shape
        self.n_rows = G.shape[0]
        self.n_cols = G.shape[1]
        self.n_states = self.n_cols * self.n_rows

        # State space - tile number representation
        self.labels = np.arange(1,self.n_states+1)
        self.obs = np.arange(2)

        # Setting Observation Matrix
        self.theta = theta

        # Set State Transition Matrix (and Prior Vector and Observation Matrix)
        self.P_S = np.zeros((self.n_states,self.n_states))
        self.P_1 = np.zeros(self.n_states)
        self.d_obs = len(theta)
        theta = np.array(theta)
        Theta = np.vstack((1 - theta, theta)).T
        self.P_O = np.zeros((self.n_states,self.d_obs,2))
        for s in range(self.n_states):
            i,j = divmod(s, self.n_cols) 
            if self.G[i,j] > 3:
                self.P_1[s] = 1
            # By default, P(0 = 0) = 1
            self.P_O[s,0] = [1,0]
            self.P_O[s,1] = [1,0]
            if fps:
                # Allow for false positives
                self.P_O[s,0] = [0.9,0.1]
                self.P_O[s,1] = [0.9,0.1]
            if self.G[i,j] == 2 or self.G[i,j] == 3:
                self.P_O[s,0] = Theta[0]
            if self.G[i,j] == 1 or self.G[i,j] == 3:
                self.P_O[s,1] = Theta[1]
            if i > 0:
                self.P_S[s,s-self.n_cols] = 1
            if i < self.n_rows - 1:
                self.P_S[s,s+self.n_cols] = 1
            if j > 0:
                self.P_S[s,s-1] = 1
            if j < self.n_cols - 1 and s < self.n_states - 1:
                self.P_S[s,s+1] = 1
            self.P_S[s,:] = self.P_S[s,:] / sum(self.P_S[s,:])

            assert(sum(self.P_S[s]) == 1)
            assert(sum(self.P_O[s,0]) == 1)
            assert(sum(self.P_O[s,1]) == 1)
        self.P_1 = self.P_1 / sum(self.P_1)

    def rwd(self, s, a):
        '''
            Reward function r(s, a) of taking action a when in state s

            Parameters
            ----------
            s : int
                true state (tile which containts the object)
            a : int
                estimated state

            Returns
            -------
            float
                reward obtained from taking action a given state s
        '''
        return (s==a)*1.

    def step(self, _s=None):
        ''' Step to the state, given prev state _s.

            Paramaters
            ----------

            _s : int
                prev state

            Returns
            -------

            s : int
                next state
            o : array_like(int, ndim=1) of shape (2)
                corresponding observation
        '''

        # Generate a state s' ~ p( . | s)

        if _s is None:
            s = np.random.choice(self.n_states,p=self.P_1)
        else:
            s = np.random.choice(self.n_states,p=self.P_S[_s,:])

        # Generate an observation o' ~ p(. | s')
        
        o = np.zeros(self.d_obs)
        for j in range(self.d_obs):
            w = self.P_O[s,j]
            #print("P(O | s=%s) = %s" % (str(s),str(self.P_O[s,j,:])))
            o[j] = np.random.choice(self.d_obs,p=w)

        return s, o


    def tile2cell(self, s):
        return divmod(s, self.n_cols) 

    def render(self, y_seq=None, x_seq=None, dgrid=None, a_star=None, paths=[], title=None, add_legend=True, output_fname=None):
        '''
            Plot a visual representation of the environment.

            Parameters
            ----------

            y_seq : numpy array (dtype=int)
                a path (e.g., [1,3,1,2])

            x_seq :
                observations associated with the path

            dgrid : shape like self.G
                contains values (e.g., probabilities) to show in each tile

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
            im = ax.imshow(self.G, cmap=ListedColormap(list(colors.values())), alpha=0.3)
            patches = [mpatches.Patch(color=colors[i], alpha=0.3, label=labels[i]) for i in [1,2,3,4]]
            if add_legend:
                plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        else:
            # ... as a probability mass function
            im = ax.imshow(dgrid.reshape(self.n_rows,self.n_cols), cmap=plt.cm.Reds)

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

        ax.set_xticks(np.arange(0, self.n_cols, 1))
        ax.set_xticks(np.arange(-0.5, self.n_cols, 1), minor=True)
        ax.set_xticklabels(np.arange(0, self.n_cols, 1))

        ax.set_yticks(np.arange(0, self.n_rows, 1))
        ax.set_yticks(np.arange(-0.5, self.n_rows, 1), minor=True)
        ax.set_yticklabels(np.arange(0, self.n_rows, 1))

        ax.grid(which='minor', color='k')

        n = 0
        for i in range(self.n_rows):
            for j in range(self.n_cols):
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
