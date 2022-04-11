import sys
import matplotlib
if sys.platform=='darwin':
    matplotlib.use("TKAgg")
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
from colors import color
import sklearn as sl
import skimage
from sklearn.metrics import mean_squared_error as MSE
from data import Sample, smooth, Spectra
from scipy.optimize import basinhopping
from scipy.integrate import trapz
from camera import CameraCalibration

from tmm.tmm_core import coh_tmm, unpolarized_RT

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

#operational wavelenghts on which we are doing regression
nfiles=28
wavels = np.linspace(405, 740, 70)
filename_templates = ['Sample_%02d' %  i for i in range(1,nfiles)]
generate=False
samples_file='ab_samples_ld.npy'
if generate:
    samples = [Sample(file, 'Mirror', spectra_kargs={'transform' : smooth}, thick_file='thicknesses.pkl')
               for file in filename_templates]
    np.save(samples_file, samples)
else:
    samples = np.load(samples_file,allow_pickle=True)

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
    # wavels = np.linspace(405, 750, 100)
    # for d in np.linspace(100, 500, 5):
    #     reflections = [get_refl(wl0, d) for wl0 in wavels]
    #     plt.plot(wavels, reflections, label='d=' + str(d))
    # plt.legend()
    # plt.show()
    camera = CameraCalibration.load('camera_ld.npy')
    dmin=[60]
    dmax=[500]
    #sample=np.random.choice(samples)
    sample=np.random.choice(samples)
    bounds = [(low, high) for low, high in zip(dmin, dmax)]
    minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds)
    #print(sample.spectra.f(750))
    # fast option, uncomment. Put in a loop with random initial point!
    # x0 = [300]
    # #res = basinhopping(lambda x: myfun(x,sample), x0, minimizer_kwargs=minimizer_kwargs)
    # # slower option but more stable
    # take_step = RandomDisplacementBounds(dmin, dmax, stepsize=[200])
    # res = basinhopping(lambda x: myfun_with_scaling(x,scaling,sample), x0, niter=30,
    #                    minimizer_kwargs=minimizer_kwargs,
    #                    take_step=take_step)
    # d0 = res.x[0]
    ds = np.linspace(dmin[0],dmax[0],200)
    cost_functions=[myfun_color(d,sample,camera) for d in ds]
    d0=ds[np.argmin(cost_functions)]
    print('estimated thickness',d0)
    print('measured thickness',sample.thickness)
    plt.figure(figsize=[10,5])
    # plt.subplot(221)
    # plt.title('estimated thickness=%.1f, measured thickness=%.1f' % (d0,sample.thickness))
    # plot_spectra(d0, color='r',label='fitted spectra')
    # spectra = sample.spectra.f(wavels)
    # #plt.plot(wavels,spectra,'b',label='raw measured spectra')
    # plot_spectra(sample.thickness,color='k',label='spectra with measured thickness')
    # plt.legend()
    # plt.xlabel('wl, nm')
    # plt.ylabel('reflection')
    #
    plt.subplot(121)
    plt.title('estimated thickness=%.1f, measured thickness=%.1f' % (d0,sample.thickness))
    plt.plot(ds,cost_functions)
    plt.axvline(d0,color='r', label='estimated thickness')
    plt.axvline(sample.thickness, color='k', label='measured thickness')
    plt.xlabel('thickness, nm')
    plt.ylabel('cost')
    #
    plt.subplot(122)
    pdf, conf = confidence(cost_functions, ds)
    plt.title('confidence score %.2f' % conf)
    plt.plot(ds,pdf)
    plt.yticks([])
    plt.axvline(d0,color='r', label='estimated thickness')
    plt.axvline(sample.thickness, color='k', label='measured thickness')
    plt.xlabel('thickness, nm')
    plt.ylabel('pdf')

    #
    # plt.subplot(234)
    # plt.title('Color from camera image')
    # sample.image.show_estimated_color()
    # plt.subplot(235)
    # plt.title('Camera image')
    # sample.image.show_true_image()
    # plt.subplot(236)
    # plt.title('Calibrated camera color (estimated thickness)')
    # intensities = [get_refl(wl, sample.thickness) for wl in wavels]
    # rgb = camera.spectra_to_XYZ(wavels,intensities)
    # Spectra.show_calculated_color(rgb)
    #plt.savefig('results/color-thickness/'+sample.filename+'.png')
    plt.show()
    #plt.close()
    estimated_thicknesses = []
    gt_thicknesses = []
    confidences = []
    for sample in samples:
        ds = np.linspace(dmin[0], dmax[0], 200)
        cost_functions = [myfun_color(d, sample, camera) for d in ds]
        d0 = ds[np.argmin(cost_functions)]
        print('estimated thickness', d0)
        print('measured thickness', sample.thickness)
        plt.figure(figsize=[10, 5])
        plt.subplot(121)
        plt.title('estimated thickness=%.1f, measured thickness=%.1f' % (d0, sample.thickness))
        plt.plot(ds, cost_functions)
        plt.axvline(d0, color='r', label='estimated thickness')
        plt.axvline(sample.thickness, color='k', label='measured thickness')
        plt.xlabel('thickness, nm')
        plt.ylabel('cost')
        #
        plt.subplot(122)
        pdf, conf = confidence(cost_functions, ds)
        plt.title('confidence score %.2f' % conf)
        plt.plot(ds, pdf)
        plt.yticks([])
        plt.axvline(d0, color='r', label='estimated thickness')
        plt.axvline(sample.thickness, color='k', label='measured thickness')
        estimated_thicknesses.append(d0)
        gt_thicknesses.append(sample.thickness)
        confidences.append(conf)
        plt.xlabel('thickness, nm')
        plt.ylabel('pdf')
        plt.savefig('new_results/color-thickness/new_set/' + sample.filename + '2.png')
        plt.close()
    estimated_thicknesses = np.array(estimated_thicknesses)
    gt_thicknesses = np.array(gt_thicknesses)
    confidences = np.array(confidences)
    plt.figure(figsize=[10,5])
    difference = np.abs(estimated_thicknesses-gt_thicknesses)
    plt.bar(np.arange(len(samples)),difference,color='b', label = 'confidence > 40%')
    mask = confidences > 0.4
    difference[mask] = 0
    plt.bar(np.arange(len(samples)), difference, color='r',label = 'confidence < 40%')
    plt.legend()
    plt.xlabel('samples')
    plt.ylabel('thick abs err')
    plt.savefig('new_results/color-thickness/new_set/gt_vs_est2.png')
    plt.close()




