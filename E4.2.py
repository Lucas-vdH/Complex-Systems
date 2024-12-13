import networkx as nx
import numpy as np
import scipy.sparse.linalg
from sklearn.linear_model import Ridge 
import matplotlib.pyplot as plt


# Creating Reservoir Network, which is the adjacency matrix A. Shape (Nr, Nr). Nr -> Nodes in Reservoir, all nodes connect to each other
n = 500 # network dimension
p = 0.01 # edge connection probability
seed = 52 # seed for the random generation
network = nx.fast_gnp_random_graph(n=n, p=p, seed=seed) 
network = nx.to_numpy_array(network)

network = np.array([[np.random.uniform(-1, 1) for i in network[j]] for j in range(len(network))])

s_rad = 0.4 # spectral radius
network = scipy.sparse.csr_matrix(network)
eigenvals = scipy.sparse.linalg.eigs(network, k=1, v0=np.ones(n), maxiter=int(1e3 * n))[0]
maximum = np.absolute(eigenvals).max() 
network = (s_rad / maximum) * network
network = np.array(network.todense())
print(f'Shape of reservoir network: {network.shape}')

# Creating w_in. Shape (Nr, 3)
w_in = np.random.default_rng(seed)
w_in = w_in.uniform(low=-0.05, high=0.05, size=(n, 3))
print(f'Shape of w_in: {w_in.shape}')

def next_r(r_i, x_i):
    '''Computes the next reservoir state'''
    next_r = np.tanh(network @ r_i + w_in @ x_i)
    return next_r

# Loading the data set of Shape (T, 3) and constructing the Reservoir State of Shape (N, T-2000)
X = np.load("lorenz_data.npy")
test_size = 500
x = X[:-test_size, :]
x_test = X[-test_size:, :]
print(f'Shape of x: {x.shape}')
r0 = np.zeros((n, 1))
R = r0
ri = r0
for i in range(x.shape[0]-1):
    ri = next_r(ri, x[i].reshape(-1, 1))
    R = np.append(R, ri, axis=1)    
    if i % 400 == 0:
        print(f"Iteration: {i}")
R = R.T
print(f'Shape of reservoir states: {R.shape}')
R = R[2000:, :]
print(f'Shape of reservoir states after slicing {R.shape}')

# Create Y (next state) as x shifted by one. Shape (T-2000, 3)
Y = x[1:, :]
Y = np.vstack([Y, np.zeros(x.shape[1])])
print(f'Shape of target Y: {Y.shape}')
Y = Y[2000:, :]
print(f'Shape of target Y after slicing: {Y.shape}')
# Slicing again to match shape of R, useful when wanting to test model with lower training times
Y = Y[:R.shape[0], :]
print(f'Shape of Y after slicing again: {Y.shape}')

# Training W_out. Shape (3, Nr)
regressor = Ridge(alpha=10**-2)
regressor.fit(R, Y) 
W_out = regressor.coef_
print(f'Shape of W_out: {W_out.shape}')

# Getting predictions
prediction_range = 500
pred = []
for i in range(prediction_range):
    prediction = W_out @ ri
    ri = np.tanh(w_in @ prediction + network @ ri)
    pred.append(prediction)
    
    if i % 400 == 0:
        print(f"Prediction step: {i}")

pred = np.array(pred)
print(f'Shape of Predictions: {pred.shape}')

time_steps = np.arange(pred.shape[0])

# Plotting Predicted vs Real
dimensions = ['x', 'y', 'z'] 
plt.figure(figsize=(16, 8))
x_test = x_test[:pred.shape[0], :]
print(f'Shape of x_test: {x_test.shape}')
for i in range(3):
    plt.subplot(3, 1, i + 1)
    plt.plot(time_steps, pred[:, i], label=f"Predicted {dimensions[i]}", color='blue', linewidth=0.5)
    # plt.plot(time_steps, x_test[:, i], label=f"True value", color='red', linewidth=0.5)
    plt.xlabel("Time Steps")
    plt.ylabel(f"{dimensions[i]} Value")
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.savefig(f'Predictions{n}bigalpha.png', format='png')

plt.figure(figsize=(16, 8))
for i in range(3):
    plt.subplot(3, 1, i + 1)
    plt.plot(time_steps, pred[:, i], label=f"Predicted {dimensions[i]}", color='blue', linewidth=0.5)
    plt.plot(time_steps[:x_test.shape[0]], x_test[:, i], label=f"True value", color='red', linewidth=0.5)
    plt.xlabel("Time Steps")
    plt.ylabel(f"{dimensions[i]} Value")
    plt.legend()
    plt.grid()
plt.tight_layout()
plt.savefig(f'Predictions&Real{n}bigalpha.png', format='png')

plt.show()

# Hyper Parameters
'''
Nr -> Number of nodes in Reservoir. Higher amount of nodes may capture better flexibility but are very computationally costly.
p -> Edge connection probability. Empirical. The rest of the connections are done with random values between -1 and 1
s_rad -> Specral Radius: Controls largest eigenvalue of reservoir matrix. Represents stability and prediction horizon of the reservoir
w_in value range -> How to initialize W_in (-0.05 to 0.05 in this implementation)
alpha (in Regressor) -> Strength of regularization during training of W_out
activation function (tanh) -> Activation function for nonlinearity. Convention and Empirical. 
'''