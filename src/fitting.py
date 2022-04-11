import numpy as np
from scipy.integrate import trapz as integr
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pickle
from scipy.optimize import nnls


class NNLS:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if len(y.shape) > 1:
            nchannels = y.shape[1]
        else:
            nchannels = 1
            y = y.reshape(-1, 1)
        self.nchannels = nchannels
        ### add 0 dimension
        XX = np.ones([X.shape[0], X.shape[1] + 1])
        XX[:, 1:] = X[:, :]
        if not self.fit_intercept:
            XX = X.copy()
        a = np.ones([XX.shape[1], y.shape[1]])
        for i in range(nchannels):
            a[:, i], _ = nnls(XX, y[:, i])
        if self.fit_intercept:
            self.coef_ = a[1:, :].T.copy()
            self.intercept_ = a[0, :]
        else:
            self.intercept_ = np.zeros(nchannels)
            self.coef_ = a.T.copy()

    def predict(self, X):
        return X @ self.coef_.T + self.intercept_

    def score(self, X, y):
        y_pred = self.predict(X)
        return MSE(y, y_pred)

class RBF():
    def __init__(self,x,y,nmodes,fitting='sin'):
        if fitting=='sin':
            self.t_n=self.sin_n
        elif fitting=='poly':
            self.t_n=self.poly_n
        elif fitting == 'gaussian':
            self.t_n = self.gaussian
        elif fitting == 'lorenzian':
            self.t_n = self.gaussian
        else:
            ValueError('Unrecognized function type!')
        self.nmodes = nmodes
        self.x=x
        self.y=y
        self.N = len(y)

    def sin_n(self,n):
        x = np.arange(self.N)
        return 2 * np.sin(np.pi * (n + 1) * (x + 1) / (len(x) + 1)) / 2 / (self.N + 1)

    def poly_n(self,n):
        pass

    def gaussian(self,n,sigma=10):
        N = self.N + (self.N % self.nmodes)
        xi = np.arange(0,N,N/self.nmodes)[n]
        x = np.arange(self.N)
        return np.exp(-np.abs(x-xi)**2/sigma**2)

    def lorenzian(self,n,sigma=20):
        N = self.N + (self.N % self.nmodes)
        xi = np.arange(0,N,N/self.nmodes)[n]
        x = np.arange(self.N)
        y = (x-xi)/sigma/2
        return 1/(1+y**2)

    def get_X(self):
        X = []
        for i in range(self.nmodes):
            x=self.t_n(i)
            X.append(x)
        return np.array(X).T

