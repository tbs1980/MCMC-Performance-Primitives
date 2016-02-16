import numpy as np
import mpp_module

# define the dimensionality of the Gaussian posterior
numParams = 10

# define the mean of the distribution
mu = np.zeros(numParams)

# define the invese of the diagonal covariance matrix, i.e, 1/var(i)
sigma_inv = np.ones(numParams)

# define log-posterior function, value is returned to val[0]
def logPostFunc(q):
    val = -0.5*np.sum( (mu-q)*(mu-q)*sigma_inv )
    return val

# define the derivatives of the log-post wrt q
def logPostDerivs(q):
    dq = sigma_inv*(mu-q)
    return dq

# maximum value of the parameter epsilon; 0<epsilon<2
maxEps = 1.

# maximum number of leapfrog / euler steps
maxNumSteps = 10

# starting point for the sampling
startPoint = np.zeros(numParams)

# seed for the random number generator
randSeed = 1234

# number samples to be taken in one interation.
# how often you would like the chains to be written?
# every time packet_size samples
packetSize = 100

# number samples to be burned
numBurn = 0

# number samples to be taken (after burning)
numSamples = 1000

# path to chains and other output
rootPathStr = "./testPyMPP"

# do we require output to console? 0 means NO, !=0 means YES
consoleOutput = 1

# delimiter for the chain data
delimiter = ","

#  percision with which the chains should be written
precision = 10

# inverse of the diagonal kinetic energy mass matrix
# this should be close/equal to the invese of the covariance
# matrix of parameters / posterior distriubtion if Gaussian
# If parameters are correlated, try diagonal elemens
keDiagMInv = np.ones(numParams)

# call mpp interface
mpp_module.canonicalHamiltonianSampler(
    numParams,
    maxEps,
    maxNumSteps,
    startPoint,
    randSeed,
    packetSize,
    numBurn,
    numSamples,
    rootPathStr,
    consoleOutput,
    delimiter,
    precision,
    keDiagMInv,
    logPostFunc,
    logPostDerivs
)
