import numpy as np

def sigmoidActivation(x):
    return 1/(1+np.exp(-x))
firstInputs = np.array([[1,0,0,0],
                        [0,1,0,1],
                        [1,0,1,0],
                        [1,1,0,1]])
print('first inputs are')
print(firstInputs)
trueOutputs = np.array([[1,0,1,1]]).T


np.random.seed(1)

weights = 2 * np.random.random((4,1)) - 1
print('random weights are:')
print(weights)
for i in range (30000):
    inputLayerInit = firstInputs
    output = sigmoidActivation(np.dot(inputLayerInit, weights))
    error = trueOutputs - output
    adjustments = np.dot(inputLayerInit.T, error * (output* (1-output)))
    weights+=adjustments


print('weights after learning')
print(weights)


print('exit outputs')
print (output)
