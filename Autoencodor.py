# Step 1: Import packages
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import GPy
import matplotlib.pyplot as plt
from pyDOE import lhs

# Step 2: Generated data 
def f(x,t):
    """ fonction to learn f(x,t) = (sin(4 \pi (x/4 + 1 ) t + x/10)+1)2
        The function is different from PCA to be between 0 and 1"""
    return((torch.sin( 4*math.pi*(x/4+1)*t+ x/10)*0.5+1)/2)

N, dim, Nt = 40, 1, 128
X = torch.tensor(lhs( dim, samples = N))
t = torch.linspace(0, 1, Nt).reshape((1,Nt))
Y = f( X, t)

# Step 3: Define the auto-encoder using PyTorch
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
        nn.Linear(input_dim, input_dim//2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(input_dim//2, input_dim//4),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(input_dim//4, encoding_dim)
        )
        self.decoder = nn.Sequential(
        nn.Linear(encoding_dim, input_dim//4),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(input_dim//4, input_dim//2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(input_dim//2, input_dim),
        nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Step 4: Train the auto-encoder
input_dim = Y.shape[1]
encoding_dim = 16
autoencoder = AutoEncoder(input_dim, encoding_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.0001, weight_decay = 1e-8)


autoencoder.train()
num_epochs = 100000
for epoch in range(num_epochs):
    inputs = torch.tensor(Y, dtype=torch.float32)
    targets = inputs.clone()

    outputs = autoencoder(inputs)
    loss = criterion(outputs, inputs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(epoch, loss)

# Step 5: Obtain encoded data from the auto-encoder
autoencoder.eval()
Y_latent = autoencoder.encoder(torch.tensor(Y, dtype=torch.float32)).detach().numpy()

# Step 6: Gaussian Process Regression using encoded data
GPs = []
for i in range(min(Y_latent.shape[1],10)):  # Construction of the surrogates models
    kernel = GPy.kern.RBF(input_dim=dim, ARD=True) # one kernel for all GP regression
    m = GPy.models.GPRegression( X.detach().numpy(), Y_latent[:,i].reshape((N,1)),kernel=kernel)
    m[".*Gaussian_noise"] = m.Y.var() * 0
    m[".*Gaussian_noise"].fix()
    m.optimize(max_iters=5000, optimizer='scg')  # optimisation of the hyperparameters
    # m.optimize_restarts(num_restarts=10, robust=True, verbose=False, parallel=True, num_processes=10) # Activate if numpy version <= 1.23.0
    GPs.append(m)   # add the GP to the great model

# Step 7: Prediction of the GP
N_test = 1000
X_test = np.random.uniform(0,1, (N_test,dim))
Y_latent_test_m = np.zeros((N_test,Y_latent.shape[1]))
Y_latent_test_s = np.zeros((N_test,Y_latent.shape[1]))
for i in range(len(GPs)):
    temp_mu, temp_v = GPs[i].predict(X_test)
    Y_latent_test_m[:,i] = temp_mu.T
    Y_latent_test_s[:,i] = temp_v.T

y_pred = autoencoder.decoder(torch.tensor(Y_latent_test_m, dtype=torch.float32)).detach().numpy()

# Step 8: Plots
def r2(y, yhat):
    if torch.var(y)<10**(-15):
        return(1)
    return( 1 - torch.mean((y-yhat)**2)/torch.var(y))

def r2func( y, yhat):
    error = torch.zeros(y.shape[1])
    for i in range(y.shape[1]):
        error[i] = r2(y[:,i], yhat[:,i])
    return(error)

plt.plot(r2func( f(torch.tensor(X_test),t), y_pred))
plt.show()

for i in range(20):
    plt.plot( y_pred[i,:], '--', color='red')
    plt.plot( f(torch.tensor(X_test),t)[i], 'red')
plt.show()

for i in range(20):
    plt.plot( f(torch.tensor(X_test),t)[i], color='red')
    plt.plot( autoencoder(torch.tensor(f(torch.tensor(X_test),t), dtype=torch.float32))[i,:].detach().numpy(),'--', color='red')
plt.show()