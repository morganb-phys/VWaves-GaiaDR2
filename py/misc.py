import numpy as np
from scipy.optimize import minimize
import emcee
import corner
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from numpy import random

class ncount():
    
    nw= 100
    nsteps= 1000
    
    def __init__(self,data,magnitude,bprp,a=0.025,guess=[1.,0.03,0.1,0.1,0.5],mcmc=False):
        
        #Unpack the input parameters
        self.x,self.y,self.z= data.T
        self.Mg= magnitude
        self.binw=a   
        self.g= guess
        self.br= bprp
        self.mcmc= mcmc
        
        #Calculate some important parameters
        self.lim= self.calc_lim() 
        if np.isnan(self.lim[1]):
            print('The colour cut ',self.br,' is not complete!')
            raise Exception()
        self.area= np.pi*0.025**2*self.binw
        self.r= [-np.round(self.lim[1],2),np.round(self.lim[1],2)]
        self.b= np.linspace(self.r[0],self.r[1],int(np.round((self.r[1]-self.r[0])/self.binw)))
        
        #Calculate the number count density with and without cuts based on magnitude
        self.zbin_uncut, self.N_uncut= self.calc_count()
        self.zbin= self.zbin_uncut[abs(self.zbin_uncut)>self.lim[0]]
        self.N= self.N_uncut[abs(self.zbin_uncut)>self.lim[0]]

        #Update guess
        self.g[0]= np.log10(self.g[0]*np.max(self.N))
        
        #Calculate the best fit to the number counts
        self.zmodel= np.linspace(-self.lim[1],self.lim[1],1000)
        self.psech= self.calc_sechfit()
        self.n0,self.zsun,self.H1,self.f,self.H2= self.psech
        self.n0= 10.**self.n0
        
        #Perform the MCMC to find the best values if required
        if mcmc:
            self.sample= self.calc_mcmc()
            self.n0_mcmc, self.z0_mcmc, self.h1_mcmc, self.f_mcmc, self.h2_mcmc= self.calc_psech_mcmc()
            self.psech_mcmc= [np.log10(self.n0_mcmc[0]),self.z0_mcmc[0],self.h1_mcmc[0],self.f_mcmc[0],self.h2_mcmc[0]]
            #self.print_mcmc_bestfit()
            #self.plot_corner()
        
        #Calculate the asymmetry parameter
        self.zA,self.A,self.A_err= self.calc_A()
   
    def plot_corner(self):
        fig = corner.corner(self.sample,labels=["$n_0$", "$z_{sun}$", "$H_1$", "$f$", "$H_2$"])
        
    def print_mcmc_bestfit(self):  
        print("""The MCMC values colour bin {0:.1f} are:
            n_0 = {1:.1e}_-{2:.1e}^+{3:.1e}
            z_0 = {4:.1e}_-{5:.1e}^+{6:.1e}
            H_1 = {7:.1e}_-{8:.1e}^+{9:.1e}
            f = {10:.1e}_-{11:.1e}^+{12:.1e}
            H_2 = {13:.1e}_-{14:.1e}^+{15:.1e}"""\
                  .format(self.br,
                          self.n0_mcmc[0],
                          self.n0_mcmc[1],
                          self.n0_mcmc[2],
                          self.z0_mcmc[0],
                          self.z0_mcmc[1],
                          self.z0_mcmc[2],
                          self.h1_mcmc[0],
                          self.h1_mcmc[1],
                          self.h1_mcmc[2],
                          self.f_mcmc[0],
                          self.f_mcmc[1],
                          self.f_mcmc[2],
                          self.h2_mcmc[0],
                          self.h2_mcmc[1],
                          self.h2_mcmc[2]))
        
    def calc_lim(self):
        mg= np.array([[7.],[17.]])
        d= 10.**((mg-self.Mg)/5.-2.)
        return np.max(d[0]),np.min([np.sqrt(np.min(d[1])**2-0.25**2),2.08])   
        
    def print_lim(self):
        print("""For a colour bin of {0:.1f} to {1:.1f}
         the sample is complete over a range of {2:.2f} to {3:.2f} kpc."""\
          .format(self.br,
                  (self.br+0.1),
                  self.lim[0],
                  self.lim[1]))
    
    def calc_count(self):
        Ncount, edges= np.histogram(self.z,bins=self.b)
        mid= np.diff(edges)/2.+edges[:-1]
        
        return mid[(mid!=0)*(Ncount!=0)], Ncount[(mid!=0)*(Ncount!=0)]
    
    def nloglikelihood(self,params,data):
        N,z= data
        model= self.n_model(params,z)
        if (params[4]>5. or params[4]<0):
            return np.inf
        if (params[3]>10. or params[3]<0.):
            return np.inf
        if (params[2]<0. or params[2]>5.):
            return np.inf
        if (params[1]<-0.1 or params[1]>0.1):
            return np.inf
        loglike= -model+N*np.log(model)
        return -np.sum(loglike)
        
    def n_model(self,params,zdata):
        ln_n0,zsun,H1,f,H2 = params
        n0= 10.**(ln_n0)
        return n0*(1./np.cosh((zdata+zsun)/(2.*H1))**2+f*1./np.cosh((zdata+zsun)/(2.*H2))**2)
    
    def n_model_simple(self,params,zdata):
        ln_n0,zsun,H1 = params
        n0= 10.**(ln_n0)
        return n0*(1./np.cosh((zdata+zsun)/(2.*H1))**2)

    def calc_sechfit(self):
        fit= minimize(lambda x: self.nloglikelihood(x,[self.N,self.zbin]),self.g)
        return fit.x
    
    def calc_mcmc(self):
        ndim, nwalkers= len(self.psech), self.nw
        pos= [self.psech+1e-3*np.random.randn(ndim) for i in range(nwalkers)]
        
        sampler= emcee.EnsembleSampler(nwalkers,ndim,lambda x:-self.nloglikelihood(x,[self.N,self.zbin]))
        sampler.run_mcmc(pos,self.nsteps)

        samples= sampler.chain[:,(round(self.nsteps*0.1)):,:].reshape((-1,ndim))
        samples[:,0]= 10.**samples[:,0]
        
        return samples
    
    def calc_psech_mcmc(self):
        n0_mcmc,z0_mcmc,h1_mcmc,f_mcmc,h2_mcmc= map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                                         zip(*np.percentile(self.sample, [16, 50, 84],
                                                                            axis=0)))
        return n0_mcmc, z0_mcmc, h1_mcmc, f_mcmc, h2_mcmc
    
    def calc_A(self,z0=np.nan):
        if np.isnan(z0):
            z0=self.zsun
        zpos= self.z+z0
        count,edges= np.histogram(zpos,bins=self.b)
        mbin= np.diff(edges)/2.+edges[:-1]
        count= count[(abs(mbin)>self.lim[0])]/self.area
        Asym= (count-count[::-1])/(count+count[::-1])
        mbin= mbin[(abs(mbin)>self.lim[0])]
        Asym_err= np.sqrt(2.*self.N*self.N[::-1]/(self.N+self.N[::-1])**3)
        return mbin[mbin>0.], Asym[mbin>0.], Asym_err[mbin>0.]
    
    def shift_N(self,z0):
        zpos= self.z+z0
        count,edges= np.histogram(zpos,bins=self.b)
        mbin= np.diff(edges)/2.+edges[:-1]
        return count,mbin
        
    def plot_xyz2D(self):
        figsize(17,5)
        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.scatter(self.x, self.y, s=0.2)
        ax1.set_xlabel('X (kpc)'); ax1.set_ylabel('Y (kpc)')
        ax2.scatter(self.x, self.z, s=0.2)
        ax2.set_xlabel('X (kpc)'); ax2.set_ylabel('Z (kpc)')
        ax2.axhline(-self.lim[0],c='k',ls='--',lw=0.8)
        ax2.axhline(self.lim[0],c='k',ls='--',lw=0.8)
        ax2.axhline(-self.lim[1],c='k',ls='--',lw=0.8)
        ax2.axhline(self.lim[1],c='k',ls='--',lw=0.8)
        ax3.scatter(self.y, self.z, s=0.2)
        ax3.set_xlabel('Y (kpc)'); ax3.set_ylabel('Z (kpc)')
        ax3.axhline(-self.lim[0],c='k',ls='--',lw=0.8)
        ax3.axhline(self.lim[0],c='k',ls='--',lw=0.8)
        ax3.axhline(-self.lim[1],c='k',ls='--',lw=0.8)
        ax3.axhline(self.lim[1],c='k',ls='--',lw=0.8)
        
    def plot_ncount_uncut(self):
        figsize(12,6)
        figure()
        plt.yscale('symlog')
        step(self.zbin_uncut,self.N_uncut,where='mid')
        axvline(-self.lim[0],c='k',ls='--',lw=0.8); axvline(self.lim[0],c='k',ls='--',lw=0.8)
        axvline(-self.lim[1],c='k',ls='--',lw=0.8); axvline(self.lim[1],c='k',ls='--',lw=0.8)
        ylabel('N')
        xlabel('z (kpc)')
  
    def plot_sechfit(self):
        figsize(10,10)
        
        fig1= figure()
        frame1= fig1.add_axes((.1,.5,.8,.5))
        plt.yscale('symlog')
        errorbar(self.zbin,self.N,yerr=np.sqrt(self.N),fmt='o',label='data')
        if self.mcmc:
            plot(self.zmodel,self.n_model(self.psech_mcmc,self.zmodel),'k',label='best fit')
        else:
            plot(self.zmodel,self.n_model(self.psech,self.zmodel),'k',label='best fit')
        plot(self.zmodel,self.n_model(self.g,self.zmodel),\
             'k--',lw=0.8,label='initial')
        ylabel('N')
        frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
        legend()

        frame2=fig1.add_axes((.1,.3,.8,.2))        
        plot(self.zbin,(self.N-self.n_model(self.psech,self.zbin))/self.n_model(self.psech,self.zbin),'.')
        ylabel(r'$\Delta$')
        xlabel(r'z (kpc)')
        
    def plot_A(self,single=False,style='points'):
        if single:
            figure()
        plotname= str(round(self.Mg[0],1))+' to '+str(round(self.Mg[1],1))
        if style=='lines':
            plot(self.zA,self.A,'-',label=plotname,zorder=-32)
        elif style=='dash':
            plot(self.zA,self.A,'--',label=plotname,zorder=-32)            
        else:
            plot(self.zA,self.A,'.',color=plt.cm.plasma(norm(self.br)/2.1),label=plotname,zorder=-32)
            
            
def bootstrap(x,n_resamp):
    i= random.randint(0,len(x)-1,(n_resamp,len(x)))
    x_boot= x[i]
    m_boot= np.nanmedian(x_boot,axis=1)
    m= np.mean(m_boot)
    sigma= np.std(m_boot)
    return m,sigma