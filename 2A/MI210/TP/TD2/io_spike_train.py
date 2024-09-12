# %%
import numpy as np
import h5py
import brian2 as b2
# %%
def save_spike_train_H5(file_name,numpy_array):
    """ Function that saves numpy arrays in a binary file h5
    Args:
        fileName (str): the path where the numpy array will be saved. The absolute path should be given. It should finish with '.hdf5'
        numpyArray (numpy.array): the data to be saved
    """

    f = h5py.File(file_name, "w")
    f.create_dataset('spike_train',data =numpy_array);
    f.close()

# %%
def read_spike_train_H5(file_name):
    """ Function that reads numpy arrays in a binary file hdf5
    Args:
        file_name (str): the path where the numpy array will be saved. The absolute path should be given. It should finish with '.hdf5'
    Return:
        numpy_array (numpy.array): the read data
    """
    f = h5py.File(file_name, "r")
    numpy_array = f['spike_train'][:]
    f.close()
    return numpy_array

# %%
def adapt_input(x):
    """ Function that converts numpy arrays in a brian input current
    Args:
        x (numpy array): the array
    Return:
        brian T Array
    """
    data = np.zeros((x.shape[0]+1,1))
    data[0:x.shape[0],0] = x
    if np.min(x)<0:
        print("Warning! x minimum is negative! min (x) "+np.str(np.min(x)) +  "shifting x... new min (x)" +np.str(np.min(x-np.min(x))))    
        data -=np.min(x) 
    #defining the physical constants
    unit_time =  b2.ms
    unit_amplitude = b2.mamp
    #final current
    curr = b2.TimedArray(data*1000*unit_amplitude, dt=1. * unit_time) #the *1000 is to convert to amp and back to mamp
    return curr

# %%
def extract_spike_train_from_spike_monitor(spike_monitor,tmax):
    """ Function that extracts the spike times from the spike monitor and transform them into an array 
    Args:
        spike monitor (brian spike monitor)
        tmax (int)
    Return:
        numpy array
    """
    spike_list = np.unique(np.round(np.array(spike_monitor.spike_trains()[0]*1000)).astype('int'))
    spike_train = np.zeros(int(tmax)+1)
    for t in spike_list:
        spike_train[t] = 1
    return spike_train