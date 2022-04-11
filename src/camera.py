import numpy as np
from scipy.integrate import trapz as integr
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error as MSE
from data import Sample, smooth, Spectra, SpectraTheory
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pickle
from fitting import NNLS

class TheorSample():
    def __init__(self,*pargs,features='RGB',**kargs):
        self.spectra=SpectraTheory(*pargs,**kargs)
        self.X = self.spectra.intensity
        self.Y = getattr(self.spectra.color,features)

class CameraCalibration():
    def __init__(self,samples,nmodes=5,fitting='sin', fitting_params={}, regression=LinearRegression):
        """
        :param samples: list of Sample class instances.
        :param nmodes: number of mode in decompositions
        :param fitting: basis function types
        :param fitting_params: arguments of basis functions, for instance sigma for gaussian
        :param regression: type of solving method. Usually Linear Regression or
         Non-Negative Least Squares (NNLS defined in fitting.py file).
         NNLS is useful if you need strictly positive CMF. Like XYZ CMF, for instance.
        """
        nsamples = len(samples)
        self.fitting_params=fitting_params
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
        self.nmodes=nmodes
        X = np.zeros([nsamples,nmodes])
        y = np.zeros([nsamples,3])
        self.wl =samples[0].spectra.wl
        self.N = len(self.wl)
        for i,sample in enumerate(samples):
            P = sample.X
            XYZ = sample.Y
            #W [1,nmodes]
            X[i,:]=[integr(P*self.t_n(j,**self.fitting_params),self.wl) for j in range(nmodes)]
            # same for output vector y
            y[i,:] = XYZ[:]

        self.X = X
        self.y = y
        self.model = regression()
        self.model.fit(X,y)
        self.a = self.model.coef_
        self.CMFx, self.CMFy, self.CMFz = self.get_CMF_XYZ()

    @classmethod
    def load(cls,fname='camera.npy'):
        with open(fname,'rb') as file:
            camera = pickle.load(file)
        return camera

    def get_CMF_XYZ(self):
        XYZ=[]
        for coeff in self.a:
            res = np.zeros(len(self.wl))
            for j, k in enumerate(coeff):
                res += k * self.t_n(j,**self.fitting_params)
            f=interp1d(self.wl,res)
            XYZ.append(f)

        return XYZ

    def spectra_to_XYZ(self,wl,spectra):
        X=integr(spectra*self.CMFx(wl), wl)
        Y = integr(spectra * self.CMFy(wl), wl)
        Z = integr(spectra * self.CMFz(wl), wl)
        return np.array([X,Y,Z]) + self.model.intercept_

    def show(self,show=True):
        plt.plot(self.wl, self.CMFx(self.wl), 'r',label='$X_\lambda$')
        plt.plot(self.wl, self.CMFy(self.wl), 'g',label='$Y_\lambda$')
        plt.plot(self.wl, self.CMFz(self.wl), 'b',label='$Z_\lambda$')
        plt.legend()
        plt.title('nmodes=%d' % self.nmodes)
        plt.xlabel('wl, nm')
        if show:
            plt.show()

    def score(self,samples):
        y_pred = []
        y_true = []
        for sample in samples:
            RGB_true = sample.Y
            RGB_pred = self.spectra_to_XYZ(sample.spectra.wl, sample.X)
            y_pred.append(RGB_pred)
            y_true.append(RGB_true)
        y_true=np.array(y_true)
        y_pred = np.array(y_pred)
        return MSE(y_true,y_pred)

    def sin_n(self,n):
        x = np.arange(self.N)
        return 2 * np.sin(np.pi * (n + 1) * (x + 1) / (len(x) + 1)) / 2 / (self.N + 1)

    def poly_n(self,n):
        pass

    def gaussian(self,n,sigma=180):
        N = self.N + (self.N % self.nmodes)
        xi = np.arange(0,N,N/self.nmodes)[n]
        x = np.arange(self.N)
        return np.exp(-np.abs(x-xi)**2/sigma**2)

    def lorenzian(self,n,sigma=10):
        N = self.N + (self.N % self.nmodes)
        xi = np.arange(0,N,N/self.nmodes)[n]
        x = np.arange(self.N)
        y = (x-xi)/sigma/2
        return 1/(1+y**2)

    def save(self,fname='camera.npy'):
        with open(fname,'wb') as file:
            pickle.dump(self,file)

class MetaCalibration():
    def __init__(self,samples,nmodes=5,fitting='sin',regression=LinearRegression):
        nsamples = len(samples)
        """ 
        X - coefficient matrix <Pi,CMFr/g/b>, 
        with sizes [nsamples,nmodes]
        y - output vector with sizes [nsamples,3] 3 is for XYZ
        """
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
        self.nmodes=nmodes
        self.nfeatures = samples[0].Y.__len__()
        X = np.zeros([nsamples,nmodes])
        y = np.zeros([nsamples,self.nfeatures])
        self.wl =samples[0].wl
        self.N = len(self.wl)
        for i,sample in enumerate(samples):
            P = sample.X
            Y = sample.Y
            #W [1,nmodes]
            X[i,:]=[integr(P*self.t_n(j),self.wl) for j in range(nmodes)]
            # same for output vector y
            y[i,:] = Y[:]
        self.X = X
        self.y = y
        self.model = regression()
        self.model.fit(X,y)
        self.a = self.model.coef_
        self.CMFs = self.get_CMF_XYZ()

    @classmethod
    def load(cls,fname='camera.npy'):
        with open(fname,'rb') as file:
            camera = pickle.load(file)
        return camera

    def get_CMF_XYZ(self):
        XYZ=[]
        for coeff in self.a:
            res = np.zeros(len(self.wl))
            for j, k in enumerate(coeff):
                res += k * self.t_n(j)
            f=interp1d(self.wl,res)
            XYZ.append(f)
        return XYZ

    def spectra_to_features(self,wl,spectra):
        Y=[]
        for CMF in self.CMFs:
            Y.append(integr(spectra*CMF(wl), wl))
        return np.array(Y) + self.model.intercept_

    def show(self,show=True):
        for i,CMF in enumerate(self.CMFs):
            plt.plot(self.wl, CMF(self.wl),label='$X_{%d\lambda}$' % i)
        plt.legend()
        plt.title('nmodes=%d' % self.nmodes)
        plt.xlabel('wl, nm')
        #plt.yticks([])
        if show:
            plt.show()

    def show_mode(self,n):
        coeff = self.a[n]
        positive = np.zeros(len(self.wl))
        negative = np.zeros(len(self.wl))
        for j, k in enumerate(coeff):
            if k>=0:
                positive += k * self.t_n(j)
            else:
                negative += k * self.t_n(j)
        plt.figure()
        plt.plot(self.wl,positive,label='positive')
        plt.plot(self.wl,-negative,label='negative')
        plt.plot(self.wl, positive+negative, label='result')
        plt.legend()
        plt.show()


    def score(self,samples):
        y_pred = []
        y_true = []
        for sample in samples:
            RGB_true = sample.Y
            RGB_pred = self.spectra_to_features(sample.wl, sample.X)
            y_pred.append(RGB_pred)
            y_true.append(RGB_true)
        y_true=np.array(y_true)
        y_pred = np.array(y_pred)
        return MSE(y_true,y_pred)

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

    def save(self,fname='camera.npy'):
        with open(fname,'wb') as file:
            pickle.dump(self,file)

class MetaSpectraCalibration():
    def __init__(self, samples, filters=None, nfilters=3, fitting='sin', regression=LinearRegression):
        nsamples = len(samples)
        """ 
        X - coefficient matrix <Pi,CMFr/g/b>, 
        with sizes [nsamples,nmodes]
        y - output vector with sizes [nsamples,3] 3 is for XYZ
        """
        if fitting=='sin':
            self.t_n=self.sin_n
        elif fitting=='poly':
            self.t_n=self.poly_n
        elif fitting == 'gaussian':
            self.t_n = self.gaussian
        elif fitting == 'lorenzian':
            self.t_n = self.gaussian
        else:
            ValueError('Unreckognized function type!')
        self.wl =samples[0].spectra.wl
        self.N = len(self.wl)
        if filters is None:
            self.nfilters = nfilters
            filters = np.array([interp1d(self.wl, self.t_n(j)) for j in range(self.nfilters)])
        self.nfilters = filters.shape[-1]
        ndim = filters.shape.__len__()
        self.nfeatures = samples[0].Y.__len__()
        if ndim == 1:
            #same filters for different features
            self.filters = np.empty((self.nfeatures,self.nfilters), dtype=object)
            for i in range(self.nfeatures):
                self.filters[i] = filters
        else:
            self.filters = filters
        X = np.zeros([self.nfeatures, nsamples, self.nfilters])
        y = np.zeros([self.nfeatures, nsamples])
        for i in range(self.nfeatures):
            for j,sample in enumerate(samples):
                filters = self.filters[i]
                P = sample.X
                Y = sample.Y[i]
                #W [1,nmodes]
                X[i,j,:]=[integr(P*filter(self.wl),self.wl) for filter in filters]
                # same for output vector y
                y[i,j] = Y
        self.X = X
        self.y = y
        self.models = [regression() for _ in range(self.nfeatures)]
        for i in range(self.nfeatures):
            self.models[i].fit(X[i],y[i])
        self.a = [model.coef_[0] for model in self.models]
        self.intercept_ = np.array([model.intercept_[0] for model in self.models])
        self.CMFs = self.get_CMF_XYZ()

    @classmethod
    def load(cls,fname='camera.npy'):
        with open(fname,'rb') as file:
            camera = pickle.load(file)
        return camera

    def get_CMF_XYZ(self):
        XYZ=[]
        for i, coeff in enumerate(self.a):
            res = np.zeros(len(self.wl))
            for j, k in enumerate(coeff):
                res += k * self.filters[i,j](self.wl)
            f=interp1d(self.wl,res)
            XYZ.append(f)
        return XYZ

    def spectra_to_features(self,wl,spectra):
        Y=[]
        for CMF in self.CMFs:
            Y.append(integr(spectra*CMF(wl), wl))
        return np.array(Y) + self.intercept_

    def show(self,show=True):
        for i,CMF in enumerate(self.CMFs):
            plt.plot(self.wl, CMF(self.wl),label='$X_{%d\lambda}$' % i)
        plt.legend()
        plt.title('nmodes=%d' % self.nfilters)
        plt.xlabel('wl, nm')
        #plt.yticks([])
        if show:
            plt.show()

    def show_mode(self,n):
        coeff = self.a[n]
        positive = np.zeros(len(self.wl))
        negative = np.zeros(len(self.wl))
        for j, k in enumerate(coeff):
            if k>=0:
                positive += k * self.filters[n,j](self.wl)
            else:
                negative += k * self.filters[n,j](self.wl)
        plt.figure()
        plt.plot(self.wl,positive,label='positive')
        plt.plot(self.wl,-negative,label='negative')
        plt.plot(self.wl, positive+negative, label='result')
        plt.legend()
        plt.show()

    def score(self,samples):
        y_pred = []
        y_true = []
        for sample in samples:
            RGB_true = sample.Y
            RGB_pred = self.spectra_to_features(self.wl, sample.X)
            y_pred.append(RGB_pred)
            y_true.append(RGB_true)
        y_true=np.array(y_true)
        y_pred = np.array(y_pred)
        return MSE(y_true,y_pred)

    def sin_n(self,n):
        x = np.arange(self.N)
        return 2 * np.sin(np.pi * (n + 1) * (x + 1) / (len(x) + 1)) / 2 / (self.N + 1)

    def poly_n(self,n):
        pass

    def gaussian(self,n,sigma=20):
        print(sigma)
        N = self.N + (self.N % self.nfilters)
        xi = np.arange(0,N,N/self.nfilters)[n]
        x = np.arange(self.N)
        return np.exp(-np.abs(x-xi)**2/sigma**2)

    def lorenzian(self,n,sigma=10):
        N = self.N + (self.N % self.nfilters)
        xi = np.arange(0,N,N/self.nfilters)[n]
        x = np.arange(self.N)
        y = (x-xi)/sigma/2
        return 1/(1+y**2)

    def save(self,fname='camera.npy'):
        with open(fname,'wb') as file:
            pickle.dump(self,file)

def plot_performance_vs_nmodes(number_of_modes,train,test,file=None,**kargs):
    cameras=[CameraCalibration(train,nmodes=nmodes,**kargs) for nmodes in number_of_modes]
    train_score=[]
    test_score=[]
    for camera in cameras:
        train_score.append(camera.score(train))
        test_score.append(camera.score(test))
    plt.figure(figsize=[10,5])
    plt.plot(number_of_modes,train_score,label='train')
    plt.plot(number_of_modes, test_score,label='test')
    plt.xlabel('Number of modes')
    plt.ylabel('MSE')
    plt.legend()
    if file:
        plt.savefig(file)
        plt.close()
    else:
        plt.show()

    for nmodes, camera in zip(number_of_modes, cameras):
        plt.figure()
        camera.show(show=False)
        plt.savefig('results/'+'CMF-NMODES-%d-theor.png' % nmodes)
        plt.close()

if __name__=='__main__':
    # operational wavelenghts on which we are doing regression
    # samples_file = 'samples_pca_6modes_750nm.npy'
    # samples = np.load(samples_file)
    # idx = np.arange(len(samples),dtype=np.int32)
    # train=np.random.choice(idx,20,replace=False)
    # test=np.setdiff1d(idx, train)
    # train = samples[train]
    # test=samples[test]
    # nmodes=10
    # #plot_performance_vs_nmodes(number_of_modes,train,test,file='results/performance-exp.png')
    # camera=MetaCalibration(samples,
    #                        nmodes=nmodes,
    #                        fitting='gaussian',
    #                        regression=LinearRegression)
    # print(camera.model.intercept_)
    # print(camera.a.shape,camera.wl.shape,camera.y.shape)
    # score = camera.score(samples)
    # print(score)
    # camera.show()
    # for i in range(6):
    #     camera.show_mode(i)
    #camera.save('metacam6.npy')
    ##########################################################
    # uncomment for calculating camera CMF
    datadir='data/ocean_ccd/'
    nfiles = 52
    filename_templates = ['Sample_%02d' % i for i in range(1, nfiles)]
    generate = True
    samples_file = 'samples.npy'
    if generate:
        samples = [Sample(file, 'Mirror2', folder = datadir, spectra_kargs={'transform': smooth, 'normalized' : True, 'scaling' : 0.5}, thick_file='thicknesses.pkl')
                   for file in filename_templates]
        for i,sample in enumerate(samples):
            samples[i].X = sample.spectra.intensity
            samples[i].Y = sample.image.RGB
        np.save(samples_file, samples)
        samples=np.array(samples)
    else:
        samples = np.load(samples_file)
    idx = np.arange(len(samples),dtype=np.int32)
    train=np.random.choice(idx,20,replace=False)
    test=np.setdiff1d(idx, train)
    train = samples[train]
    test=samples[test]
    ##########################################################
    # uncomment for calculating eye CMF
    # thicknesses = np.linspace(50,700,50)
    # train=np.random.choice(thicknesses,int(len(thicknesses)*0.8),replace=False)
    # test = np.setdiff1d(thicknesses, train)
    # train = [TheorSample(d) for d in train]
    # test = [TheorSample(d) for d in test]
    # samples=np.concatenate([train,test])
    ##########################################################
    #plot_performance_vs_nmodes(number_of_modes,train,test,file='results/performance-exp.png')
    camera=CameraCalibration(samples,nmodes=8,fitting='gaussian')
    camera.save()
    camera = CameraCalibration.load()
    score = camera.score(test)
    print(score)
    camera.show()
    # for sample in samples:
    #     if sample in test:
    #         stype='TEST'
    #     elif sample in train:
    #         stype='TRAIN'
    #     else:
    #         ValueError("Unreckognized type!")
    #     cameras = [CameraCalibration(train, nmodes=nmodes) for nmodes in number_of_modes]
    #     # plt.figure(figsize=[20,20])
    #     # plt.subplot(111)
    #     # plt.suptitle(stype+' sample')
    #     # for i, camera in enumerate(cameras):
    #     #     RGB=camera.spectra_to_XYZ(sample.spectra.wl,
    #     #                           sample.spectra.intensity)
    #     #     RGB_image = sample.image.RGB
    #     #     RGB_spectra = sample.spectra.color.RGB
    #     #     y=camera.model.predict(camera.X)
    #     #     cost=MSE(camera.y,y)
    #     #     plt.subplot(4,4,4*i+1)
    #     #     camera.show(show=False)
    #     #     print(np.array(RGB))
    #     #     print(RGB_image)
    #     #     plt.subplot(4,4,4*i+2)
    #     #     plt.title('Image color')
    #     #     Spectra.show_calculated_color(RGB_image)
    #     #     plt.subplot(4,4,4*i+3)
    #     #     plt.title('Spectra camera CMF')
    #     #     Spectra.show_calculated_color(np.array(RGB))
    #     #     plt.subplot(4,4,4*i+4)
    #     #     plt.title('Spectra eye CMF')
    #     #     Spectra.show_calculated_color(RGB_spectra)
    #     # plt.savefig('results/'+sample.filename+'-exp.png')
    #     # plt.close()
    ##########################################################
    # thicknesses = np.linspace(50,1800,250)
    # train=np.random.choice(thicknesses,int(len(thicknesses)*0.8),replace=False)
    # test = np.setdiff1d(thicknesses, train)
    # train = [TheorSample(d) for d in train]
    # test = [TheorSample(d) for d in test]
    # data=np.concatenate([train,test])
    # print(data.shape)
    # np.save('theordata',data)
    # data=np.load('theordata.npy')
    # number_of_modes=np.arange(2,18,2)
    # plot_performance_vs_nmodes(number_of_modes,train,test,file='results/performance-theor.png')
    # camera=CameraCalibration(train)
    # #camera.show()
    # sample = np.random.choice(test)
    # y=camera.model.predict(camera.X)
    # cost=MSE(camera.y,y)
    # print(cost)
    # RGB=camera.spectra_to_XYZ(sample.spectra.wl,
    #                       sample.spectra.intensity)
    # RGB_spectra = sample.spectra.color.RGB
    # print(np.array(RGB))
    # print(RGB_spectra)
    # plt.figure(figsize=[10,5])
    # plt.subplot(121)
    # plt.title('Spectra color camera CMF')
    # Spectra.show_calculated_color(np.array(RGB))
    # plt.subplot(122)
    # plt.title('Spectra color eye CMF')
    # Spectra.show_calculated_color(RGB_spectra)
    # plt.show()





