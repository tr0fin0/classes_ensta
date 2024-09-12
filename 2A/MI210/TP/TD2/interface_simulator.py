# %%
import brian2 as b2
import numpy as np

# %%
def get_Imin(V_rest, firing_threshold,membrane_resistance):
    """ Function that calculates the minimum current for the neuron to spike
    Args:
       V_rest (double),
       firing_threshold (double)
       membrane_resistance (double)
    Return:
        numpy array
    """
    #write your function here and erase the current return
    return 0

# %%
def make_constant_array(t_start, t_end, val):
    """ Function that generates an array filled with the same value, starting from a given index
    Args:
        t_start (int): time where the array is different from zero
        t_end (int): last time to consider
        val (double): value 
    Return:
        numpy array
    """
    #t_start and t_end in ms
    #val in mamp
    tmp_size = 1 + t_end #this is just for the simulator to consider t=0
    tmp = np.zeros(int(tmp_size))
    tmp[int(t_start):int(t_end)] = val
    return tmp


# %%
def make_my_current_array(args):
    """ Fucntion that generates an input current
    Args:
        You define
     Return:
         numpy array
     """
#write your function here and erase the current return
    return 0
    
    
    
    
    
    return my_sin
