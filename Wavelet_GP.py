#Step 1: Imports
import GPy
import numpy as np
import matplotlib.pyplot as plt
import math
from pyDOE import lhs

# Step 2: Generated data 
def f(x,t):
    """ fonction to learn f(x,t) = sin(4 \pi (x/4 + 1 ) t + x/10)"""
    return(np.where( t<x, np.sin( 2*math.pi*t)-2, np.sin( 2.3*math.pi*t)+ x+2) )

N, dim, Nt = 30, 1, 128
X = lhs( dim, samples = N)
t = np.linspace(0, 1, Nt).reshape((1,Nt))
Y = f( X, t)

# Step 3: Define the dimension reduction
import pywt
wlt = pywt.Wavelet('haar')
wlevel = pywt.dwt_max_level(len(t[0]), wlt)
waveletH1 = pywt.wavedec(Y, wlt, mode='constant', level=wlevel) # wavelet decomposition parameters
# apply the wavelet decomposition
Y_latent = np.zeros(Y.shape)
Y_latent[:,0] = np.array(waveletH1[0][0])
for i in range(1,len(waveletH1)):
    Y_latent[:,2**(i-1):2**(i)] = np.array([waveletH1[i][j] for j in range(len(waveletH1[i]))])

# Step 4: Gaussian process regression constuction
GPs = []
for i in range(min(Y_latent.shape[1],200)):  # Construction of the surrogates models
    kernel = GPy.kern.RBF(input_dim=dim, ARD=True) # one kernel for all GP regression
    m = GPy.models.GPRegression( X, Y_latent[:,i].reshape((N,1)),kernel=kernel)
    m[".*Gaussian_noise"] = m.Y.var() * 0
    m[".*Gaussian_noise"].fix()
    m.optimize(max_iters=5000)  # optimisation of the hyperparameters
    # m.optimize_restarts(num_restarts=10, robust=True, verbose=False, parallel=True, num_processes=10) # Activate if numpy version <= 1.23.0
    GPs.append(m)   # add the GP to the great model

# Step 5: Prediction of the GP
N_test = 1000
X_test = np.random.uniform(0,1, (N_test,dim))
Y_latent_test_m = np.zeros((N_test,Y_latent.shape[1]))
Y_latent_test_s = np.zeros((N_test,Y_latent.shape[1]))
for i in range(len(GPs)):
    temp_mu, temp_v = GPs[i].predict(X_test)
    Y_latent_test_m[:,i] = temp_mu.T
    Y_latent_test_s[:,i] = temp_v.T

# Step 6: Return in the time domain
waveletPredmu = [] ## mean 
waveletPredvar = [] ## variance
count = 0
for i in range(len(waveletH1)):
    size = waveletH1[i].shape[1]
    waveletPredmu.append(Y_latent_test_m[:,count:count+size])
    waveletPredvar.append(abs(Y_latent_test_s[:,count:count+size]))
    count = count + size

### transformation dans l'espace temporelle
Y_pred = pywt.waverec(waveletPredmu, wlt)
Y_pred_var = pywt.waverec(waveletPredvar, wlt)**2

# Step 7: Evaluation of the method
def r2(y, yhat):
    if np.var(y)<10**(-15):
        return(1)
    return( 1 - np.mean((y-yhat)**2)/np.var(y))

def r2func( y, yhat):
    error = np.zeros(y.shape[1])
    for i in range(y.shape[1]):
        error[i] = r2(y[:,i], yhat[:,i])
    return(error)

plt.plot(t[0], r2func(f(X_test,t),Y_pred), label='PCA-GP')
# plt.plot(t[0], r2func(f(X_test,t),np.mean(f(X_test,t),axis=0).reshape(Nt,1).repeat(N_test,axis=1).T), label='Théorical Mean')
# plt.plot(t[0], r2func(f(X_test,t),np.mean(Y,axis=0).reshape(1,Nt).repeat(N_test,axis=0)), label='Empirical Mean')
plt.legend()
plt.show()

for i in range(10):
    plt.plot( t[0], f(X_test,t)[i,:],'-',color='red')
    plt.plot( t[0], Y_pred[i,:], '--',color='red')
plt.show()

# Example of a curve with uncertainty interval
i=2
plt.plot( t[0], f(X_test,t)[i,:],'-',color='green', label ='f(x=0.11,t)')
plt.fill_between(t[0], Y_pred[i,:] + 1.96 * np.sqrt(Y_pred_var[i,:]),Y_pred[i,:] - 1.96 * np.sqrt(Y_pred_var[i,:]), color='red', alpha=0.5)
plt.plot( t[0], Y_pred[i,:], '--',color='red', label='$Z_{WGP}(x=0.11,t)$')
plt.xlabel('t', fontsize = 22)
plt.ylabel('Z(x=0.11,t)', fontsize = 22)
plt.legend()
plt.show()
