import numpy as np
import mpp_module


def logPostFunc(q,val):
    val[0] = 0.

def logPostDerivs(q,dq):
    dq = -q

numParams = 10
maxEps = 1.
maxNumSteps = 10
startPoint = np.zeros(numParams)
randSeed = 1234
packetSize = 100
numBurn = 0
numSamples = 1000
rootPathStr = "./testPyMPP"
consoleOutput = 1
delimiter = ","
precision = 10
keDiagMInv = np.ones(numParams)
