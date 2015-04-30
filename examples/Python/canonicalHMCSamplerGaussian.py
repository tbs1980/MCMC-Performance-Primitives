import ctypes
import numpy as np

num_params = 1000

mu = np.zeros(num_params,dtype=float)
sigma_inv = np.ones(num_params,dtype=float)



def log_post_func(q,val):
    val = -0.5*np.inner((mu-q),(mu-q)*sigma_inv)

def log_post_derivs(q,dq):
    dq = sigma_inv*(mu-q)
    
