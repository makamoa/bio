import sys
import matplotlib
if sys.platform=='darwin':
    matplotlib.use("TKAgg")
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from scipy.integrate import trapz
from camera import CameraCalibration
from tmm.tmm_core import coh_tmm, unpolarized_RT
import pickle

wavels = np.linspace(405, 740, 70)

def NRMSE(y_true,y_pred):
    """
    Normalized Root-Mean-Square Error (NRMSE)
    :y_true: Ground truth (correct) target values,
    :y_pred: Predicted values
    :return:
    """
    cost = np.sqrt(MSE(y_true, y_pred)) / (y_true.max()-y_true.min())
    return cost

def NRMSE2(y_true,y_pred):
    """
    Normalized Root-Mean-Square Error (NRMSE)
    :y_true: Ground truth (correct) target values,
    :y_pred: Predicted values
    :return:
    """
    cost = np.linalg.norm((y_pred - y_true)/y_true) / np.sqrt(len(y_true))
    return cost

def confidence(cost, d, sigma=15, th=0.5, normalize=False):
    conf=-np.log(np.array(cost))
    ref=(conf.max()-conf.min())*th+conf.min()
    conf-=ref
    conf[conf<0] = 0
    C=trapz(conf,d)
    i_max = np.argmax(conf)
    idx = (d>(d[i_max]-sigma)) & (d<=(d[i_max]+sigma))
    B = trapz(conf[idx],d[idx])
    norm = C
    if normalize:
        norm = 0
        for i, di in enumerate(d):
            idx = (d > (di - sigma)) & (d <= (di + sigma))
            norm += trapz(conf, d)
    return conf/norm, B/C

cost_function=MSE

mats_dir='mats/'
data_Si=np.loadtxt(mats_dir+'Si.txt')
data_PMMA=np.loadtxt(mats_dir+'PMMA_nam_dispersion.txt')
wl, n, k = data_Si[:,0], data_Si[:,1], data_Si[:,2]
Si_n= interp1d(wl, n)
Si_k= interp1d(wl, k)
wl, n = data_PMMA[:,0], data_PMMA[:,1]
PMMA_n= interp1d(wl, n)

def PMMA_n_extr(wl0):
    # Problem: PMMA refractive index defined only from 405nm. Solution:
    try:
        x = PMMA_n(wl0)
    except ValueError:
        x = PMMA_n(405.0)
    return x

def get_refl(wl0,d,n=PMMA_n_extr):
    nk_list = [1,n(wl0),Si_n(wl0)+1j*Si_k(wl0),1]
    d_list=[np.inf,d,20000,np.inf]
    return coh_tmm('s', nk_list, d_list, 0, wl0)['R']

def plot_spectra(d, **kargs):
    intensities = [get_refl(wl,d) for wl in wavels]
    plt.plot(wavels, intensities, **kargs)

def myfun(d, sample):
    y_pred = np.array([get_refl(wl,d) for wl in wavels])
    y_true = sample.spectra.f(wavels)
    return cost_function(y_true,y_pred)

def myfun_with_scaling(d, scaling, sample):
    y_pred = np.array([get_refl(wl,d) for wl in wavels])
    y_true = sample.spectra.f(wavels)*scaling
    return cost_function(y_true,y_pred)

def myfun_color(d, sample, camera:CameraCalibration):
    spectra = np.array([get_refl(wl, d) for wl in wavels])
    y_pred = camera.spectra_to_XYZ(wavels,spectra)
    y_true = sample.image.RGB
    return cost_function(y_true, y_pred)

def myfun_dispersion_color(d, n, sample, camera:CameraCalibration):
    spectra = np.array([get_refl(wl, d, lambda x: n) for wl in wavels])
    y_pred = camera.spectra_to_XYZ(wavels,spectra)
    y_true = sample.image.RGB
    return cost_function(y_true, y_pred)

class RandomDisplacementBounds():
    """random displacement with bounds"""
    def __init__(self, xmin, xmax, stepsize=[200]):
        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize

    def __call__(self, x):
        """take a random step but ensure the new position is within the bounds"""
        while True:
            for i, stepsize in enumerate(self.stepsize):
                x[i] += np.random.uniform(-stepsize, stepsize)
            if np.all(x < self.xmax) and np.all(x > self.xmin):
                break
        return x

if __name__=='__main__':
    pass
    # wavels = np.linspace(405, 750, 100)
    # for d in np.linspace(100, 500, 5):
    #     reflections = [get_refl(wl0, d) for wl0 in wavels]
    #     plt.plot(wavels, reflections, label='d=' + str(d))
    # plt.legend()
    # plt.show()




