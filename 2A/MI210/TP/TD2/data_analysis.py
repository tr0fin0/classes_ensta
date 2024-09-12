# 
import numpy as np
import matplotlib.pyplot as plt

# 
#################### Data Analysis Functions  ########################


####### Question 2)
def get_average_spike_rate(spike_train):
    """ Function that estimates de mean spike rate 
    Args:
        spike train (numpy array)
    Return:
        mean spike rate (double)
    """
    #write your function here and erase the current return
    return 0 

####### Question 3)
def get_isi(spike_train):
    """ Function that calculates the difference between spike times
    Args:
        spike train (numpy array)
    Return:
            [histogram values and bin edges
    """
#write your function here and erase the current return
    return 0
    
####### Question 4)
def get_autocorrelation(spike_train, max_lag):
    """ Function that estimates de autocorrelation fro 0 to a given lag
    Args:
        spike train (numpy array)
        max lag (int)
    Return:
        autocorrelation (numpy array)
    """
#write your function here and erase the current return
    return 0

	


# %%
######################################################

# %%
def plot_spike_train(spike_train): #feel free to polish this function
    """ Function that plots the spike_train
    Args:
        spike train (numpy array)
        exercise_folder_path (string): complete path to your folder
        fig_name (string): figure name. Should end with an image extension, typically .png or .svg
    """
    if np.sum(spike_train) == 0:
        print("Warning! The spike train has zero spikes!")
    fig = plt.figure(figsize = [6,4])
    ax = fig.add_axes((0.1, 0.12, 0.8, 0.8))
    spike_times = np.argwhere(spike_train)
    for spike in spike_times:
        plt.axvline(x = spike)
    plt.xlim([0,spike_train.size])
    plt.ylim([0,1])
    plt.yticks([])
    plt.xlabel('Time (ms)')
    #plt.savefig(exercise_folder_path+fig_name)
    
    

# %%
def plot_autocorrelation(norm_autocorrelation,lag):
    fig = plt.figure()
    plt.plot(norm_autocorrelation)
    plt.xlabel('Time lag')
    plt.ylabel('Autocorrelation')
    locs, labels = plt.xticks()
    plt.xticks(locs[1:-1],(locs[1:-1]-lag).astype('int'))

# %%
def plot_isi(hist_vals, bin_edges):
    plt.figure()
    plt.bar(bin_edges[:-1], hist_vals,align = 'edge',width=(bin_edges[1]-bin_edges[0]))
    plt.xlabel('Time difference between consecutive spikes')
    plt.ylabel('Probability')
    plt.ylim([0,1.2*np.max(hist_vals)])
    
