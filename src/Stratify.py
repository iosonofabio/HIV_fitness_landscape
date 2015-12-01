# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:49:36 2015

@author: vpuller
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys, os


sys.path.append('/ebio/ag-neher/share/users/vpuller/HIVEVO/HIVEVO_access') 
from hivevo.patients import Patient

#sys.path.append('/ebio/ag-neher/share/users/vpuller/myTOOLS/') 
#import Vadim_toolbox_file as vp

h = 10**(-8)
bases=np.array(['A','C','G','T','-'])

def chi2_traj(s_mu,pp,tt):
    '''Chi-squared expression for fitting the exponential trajectory into the data
    '''
    p_theory = s_mu[1]/s_mu[0]*(1-np.exp(-s_mu[0]*tt))
    
    return np.sum((p_theory - pp)**2)

def chi2_traj_p0(s_mu_p0,pp,tt):
    '''Chi-squared expression for fitting the exponential trajectory into the data
    '''
    p_theory = s_mu_p0[1]/s_mu_p0[0]*(1-np.exp(-s_mu_p0[0]*tt)) +\
    s_mu_p0[2]*np.exp(-s_mu_p0[0]*tt)
    
    return np.sum((p_theory - pp)**2)

def logLike_2nucs_exp_smuD(s_mu_D,p_k,t_k):
    '''log-likelihood for the model with exponential decay, dependent on s, mu, D'''
    s = s_mu_D[0]
    mu = s_mu_D[1]
    D0 = s_mu_D[2]
    
    dt_k = np.diff(t_k)
    d_k = (1-np.exp(-2*s*dt_k))/(2*s)
        
    L = (.5*np.log(D0*d_k)).sum() + ((p_k[1:] -(1-np.exp(-s*dt_k))*mu/s -\
    p_k[:-1]*np.exp(-s*dt_k))**2/(2*D0*d_k)).sum()
    
    return L

def amoeba_vp(func,x0,args = (),Nit = 10**4,a = None,tol_x = 10**(-4),tol_f = 10**(-4),return_f = False):
    '''Home-made realization fo Nelder-mead minimum search
        
    Input arguments:
    func - function to minimize (function fo a vecor argument of dimension no less than 2) 
    x0 - initial minimum guess
    args - additional arguments for the function
    Nit - maximum number of iterations
    a - edge length of the initial simplx
    tol - required relative tolerance
    return_f - return the function value and the number of iterations
    
    Output arguments:
    position of the minimum
    '''
    
    '''Nelder-Mead parameters'''
    alpha = 1.
    gamma = 2.
    rho = -.5
    sigma = .5
    
    n = x0.shape[0]   
    if a is None:
        a = np.ones(n)
#    a = 1. # edge length of the initial simplex
    xx = np.tile(x0,(n+1,1))
#    xx[1:,:] += a*np.eye(n)
    xx[1:,:] += np.diag(a)
    ff = np.array([func(x,*args) for x in xx])
    
    for j in xrange(Nit):
##        print np.max(np.array([LA.norm(x-xx[0,:]) for x in xx[1:,:]]))
#        if np.max(np.array([LA.norm(x-xx[0,:]) for x in xx[1:,:]])) < tol:
#            break
##        else:
##            print np.array([LA.norm(x-xx[0,:]) for x in xx[1:,:]])
#        xcenter = np.mean(xx,axis=0) 
#        if np.max(np.array([np.abs(x-xcenter) for x in xx])) < tol:
#            break
        xcenter = np.tile(np.mean(xx,axis=0),(n+1,1))
        fmean = np.mean(ff)
#        if np.max(np.abs(xx-xcenter)) < tol:
#            break
        if (np.abs(xx-xcenter) < tol_x*np.abs(xcenter)).all() and (np.abs(ff-fmean) < tol_f*np.abs(fmean)).all():
            break
        '''order'''
        jjsort = np.argsort(ff)
        ff = ff[jjsort]
        xx = xx[jjsort,:]
#        print j,'\n',xx,'\nf = ', ff
        
        '''centroid point'''
        xo = np.mean(xx[:n,:],axis=0)        
#        if np.max(np.abs(xo-xx[n,:])) < tol:
#            break
        
        '''reflection'''
        xr = xo + alpha*(xo - xx[n,:])
        fr = func(xr,*args)
#        print 'xr,fr = ', xr,fr
        if fr >= ff[0] and fr < ff[-2]:
            xx[-1,:] = xr
            ff[-1] = fr
            continue
        elif fr < ff[0]:
            xe = xo + gamma*(xo - xx[n,:])
            fe = func(xe,*args)
#            print 'xe,fe = ', xe,fe
            if fe < fr:
                xx[-1,:] = xe
                ff[-1] = fe
                continue
            else:
                xx[-1,:] = xr
                ff[-1] = fr
                continue
        else:
            xc = xo + rho*(xo-xx[n,:])
            fc = func(xc,*args)
#            print 'xc,fc = ', xc,fc
            if fc < ff[-1]:
                xx[-1,:] = xc
                ff[-1] = fc
                continue
            else:
                xx = xx[0,:] + sigma*(xx - xx[0,:])
    
    if j == Nit -1:
        print 'WARNING from amoeba_vp:\n    the maximum number of iterations has been reached, max(dx/x) = ',\
        np.max(np.abs((xx-xcenter)/xcenter))
#        np.max(np.array([LA.norm(x-xx[0,:]) for x in xx[1:,:]])) 
    if return_f:
        return xx[0,:],ff[0], j
    else:
        return xx[0,:]
        
if __name__=="__main__":
    '''Studying fluctuations of nucleotides with high fitness'''
    plt.ioff()
    patient_name = 'p1'
    gen_region = 'pol' #'gag' #'pol' #'gp41' #'gp120' #'vif' #'RRE'
    PAT = Patient.load(patient_name)
    
    outdir_name = '/ebio/ag-neher/share/users/vpuller/Fabio_data_work/Stratify/'
    if not os.path.exists(outdir_name):
        os.makedirs(outdir_name)
        
    tt = PAT.times()
    nt = tt.shape[0]
    counts = PAT.get_allele_count_trajectories(gen_region)
    freqs = PAT.get_allele_frequency_trajectories(gen_region).data
    
    L = freqs.shape[2] # genetic region length
      
    nuc1 = 'A'
    nuc2 = 'G'
    j1 = np.where(bases == nuc1)[0][0]
    j2 = np.where(bases == nuc2)[0][0]
    
    '''Characterizing sites by the number of nucleotide states with 
    non-zero probability'''
    netfreqs = freqs.sum(axis=0)
    relevant_nucs = np.zeros(netfreqs.shape,dtype = 'int')
    relevant_nucs[np.where(netfreqs > 0)] = 1
    print '# of sites with n = 1,2,3,4 nucs appearing\n', [np.count_nonzero(relevant_nucs[:4,:].sum(axis=0) == j) for j in xrange(1,5)]
    
    jj2 = np.where((relevant_nucs[j1,:] == 1)*(relevant_nucs[j2,:] == 1)*\
    (relevant_nucs[:4,:].sum(axis=0) == 2))[0]
    jjnuc0 = np.argmax(freqs[0,:4,:],axis=0)
    jjnuc0 = np.array([j1 if p[j1] < p[j2] else j2 for p in freqs[0,:4,:].T])
    

    t_k = np.zeros(tt.shape[0]+1)
    t_k[1:] = np.copy(tt)
    p_cutoff = 10**(-5)*tt[-1] 
    jj_sample = jj2[np.where(np.max(freqs[:,jjnuc0[jj2],jj2],axis=0) < p_cutoff)]
    s_mu = np.zeros((len(jj_sample),2))
    s_mu_p = np.zeros((len(jj_sample),3))
    s_mu_D = np.zeros((len(jj_sample),3))
    for idx, p in enumerate(freqs[:,jjnuc0[jj_sample],jj_sample].T):
#    p_ave = freqs[:,jjnuc0[jj_sample],jj_sample].mean(axis=1)
        s_mu0 = h*np.array([-1,1])
        s_mu_chi = amoeba_vp(chi2_traj,s_mu0,args=(p,tt),a = .001*np.array([1,1]),return_f = True)#,tol_x = h,tol_f = h)
        s_mu[idx,:] = s_mu_chi[0]
        
        s_mu_p0 = h*np.array([-1,1,0])
        s_mu_p_chi = amoeba_vp(chi2_traj_p0,s_mu_p0,args=(p,tt),\
        a = .001*np.array([1,1,1]),return_f = True)#,tol_x = h,tol_f = h)
        s_mu_p[idx,:] = s_mu_p_chi[0]
        
        p_k = np.zeros(tt.shape[0]+1)
        p_k[1:] = np.copy(p)     
        s_mu_D0 = h*np.array([1,1,1])
        s_mu_D_L = amoeba_vp(logLike_2nucs_exp_smuD,s_mu_D0,args=(p_k,t_k),\
        a = .001*np.array([1,1,1]),return_f = True)#,tol_x = h,tol_f = h)
        s_mu_D[idx,:] = s_mu_D_L[0]

    '''Fitness histogram'''
    s_cutoff = .1
    plt.figure(20)
    plt.clf()
    plt.hist(s_mu[np.where(s_mu[:,0] < s_cutoff)[0],0],bins = 50,alpha = 0.5)
    plt.hist(s_mu_p[np.where(s_mu_p[:,0] < s_cutoff)[0],0],bins = 50,alpha = 0.5)
    plt.hist(s_mu_D[np.where(s_mu_D[:,0] < s_cutoff)[0],0],bins = 50,alpha = 0.5)
    plt.xlabel('s',fontsize = 18)
    plt.legend(('chi2','p0','D'),fontsize = 18)
    plt.savefig(outdir_name + gen_region + '_'+nuc1+nuc2 + '_s_hist.pdf')
    plt.close(20)
    
    
    '''Average trajectories'''
    jj_ave = jj_sample[np.where(s_mu[:,0] >0.)]
    p_ave = freqs[:,jjnuc0[jj_ave],jj_ave].mean(axis=1)
    s_mu_chi = amoeba_vp(chi2_traj,h*np.array([-1,1]),args=(p_ave,tt),a = .001*np.array([1,1]),return_f = True,tol_x = h,tol_f = h)
    s_mu_ave = s_mu_chi[0]
    p_theory = s_mu_ave[1]/s_mu_ave[0]*(1-np.exp(-s_mu_ave[0]*tt))
    
    jj_ave_p = jj_sample[np.where(s_mu_p[:,0] >0.)]
    p_ave_p = freqs[:,jjnuc0[jj_ave_p],jj_ave_p].mean(axis=1)
    s_mu_p0_chi = amoeba_vp(chi2_traj_p0,h*np.array([-1,1,0]),args=(p_ave_p,tt),\
    a = .001*np.array([1,1,1]),return_f = True,tol_x = h,tol_f = h)
    s_mu_p_ave = s_mu_p0_chi[0]
    p_theory_p = s_mu_p_ave[1]/s_mu_p_ave[0]*(1-np.exp(-s_mu_p_ave[0]*tt)) +\
    s_mu_p_ave[2]*np.exp(-s_mu_p_ave[0]*tt)
    
    
    jj_ave_D = jj_sample[np.where(s_mu_D[:,0] >0.)]
    p_ave_D = freqs[:,jjnuc0[jj_ave_D],jj_ave_D].mean(axis=1)
    p_k = np.zeros(tt.shape[0]+1)
    p_k[1:] = np.copy(p_ave_D)     
    s_mu_D0 = h*np.array([1,1,1])
    s_mu_D_L = amoeba_vp(logLike_2nucs_exp_smuD,s_mu_D0,args=(p_k,t_k),\
    a = .001*np.array([1,1,1]),return_f = True,tol_x = h,tol_f = h)
    s_mu_D_ave = s_mu_D_L[0]
    p_theory_D = s_mu_D_ave[1]/s_mu_D_ave[0]*(1-np.exp(-s_mu_D_ave[0]*tt))
    
    plt.figure(20,figsize=(20,5))
    plt.clf()
    plt.plot(tt,p_ave)
    plt.plot(tt,p_ave_p)
    plt.plot(tt,p_ave_D)
    plt.plot(tt,p_theory)
    plt.plot(tt,p_theory_p)
    plt.plot(tt,p_theory_D)
    plt.legend(('chi2','p0','D','chi2, theory','p0, theory','D, theory'),fontsize = 18,loc=0)
    plt.xlabel('t',fontsize = 18)
    plt.savefig(outdir_name + gen_region + '_'+nuc1+nuc2 + '_p_ave.pdf')
    plt.close(20)
    

    '''Trajectories'''
    plt.figure(20,figsize=(20,15))
    plt.clf()

    plt.subplot(3,1,1)
    for jnuc in jj_ave:
        plt.plot(tt,freqs[:,jjnuc0[jnuc],jnuc])
    plt.xlabel('t',fontsize = 18)
    plt.ylabel('frequency',fontsize = 18)
    plt.title('chi2',fontsize = 18)
    
    plt.subplot(3,1,2)
    for jnuc in jj_ave_p:
        plt.plot(tt,freqs[:,jjnuc0[jnuc],jnuc])
    plt.xlabel('t',fontsize = 18)
    plt.ylabel('frequency',fontsize = 18)
    plt.title('p0',fontsize = 18)

    plt.subplot(3,1,3)
    for jnuc in jj_ave_D:
        plt.plot(tt,freqs[:,jjnuc0[jnuc],jnuc])
    plt.xlabel('t',fontsize = 18)
    plt.ylabel('frequency',fontsize = 18)
    plt.title('D',fontsize = 18)
    
    plt.savefig(outdir_name + gen_region + '_'+nuc1+nuc2 + '_traj.pdf')
    plt.close(20)
    
    
    plt.figure(200,figsize=(30,10*nt))
    plt.clf()
    for jt, t in enumerate(tt):
        plt.subplot(nt,3,3*jt+1)
        plt.hist(freqs[jt,jjnuc0[jj_ave],jj_ave] - p_ave[jt],bins=20)
        plt.xlabel('p(t) - <p(t)>',fontsize = 18)
        plt.ylabel('t = ' + str(t),fontsize = 18)
        plt.title('chi2',fontsize = 18)
        plt.subplot(nt,3,3*jt+2)
        plt.hist(freqs[jt,jjnuc0[jj_ave_p],jj_ave_p] - p_ave_p[jt],bins=20)
        plt.xlabel('p(t) - <p(t)>',fontsize = 18)
        plt.title('p0',fontsize = 18)
        plt.subplot(nt,3,3*jt+3)
        plt.hist(freqs[jt,jjnuc0[jj_ave_D],jj_ave_D] - p_ave_D[jt],bins=20)
        plt.xlabel('p(t) - <p(t)>',fontsize = 18)
        plt.title('D',fontsize = 18)
    plt.savefig(outdir_name + gen_region + '_'+nuc1+nuc2 + '_noise.pdf')
    plt.close(200)
    

    
#    '''Trajectories for the negative s values'''
#    jj_ave = jj_sample[np.where(s_mu[:,0] <0.)]
#    p_ave = freqs[:,jjnuc0[jj_ave],jj_ave].mean(axis=1)
#    s_mu_chi = vp.amoeba_vp(chi2_traj,h*np.array([-1,1]),args=(p_ave,tt),a = .001*np.array([1,1]),return_f = True,tol_x = h,tol_f = h)
#    s_mu_ave = s_mu_chi[0]
#    p_theory = s_mu_ave[1]/s_mu_ave[0]*(1-np.exp(-s_mu_ave[0]*tt))
#    
#    jj_ave_p = jj_sample[np.where(s_mu_p[:,0] <0.)]
#    p_ave_p = freqs[:,jjnuc0[jj_ave_p],jj_ave_p].mean(axis=1)
#    s_mu_p0_chi = vp.amoeba_vp(chi2_traj_p0,h*np.array([-1,1,0]),args=(p_ave_p,tt),\
#    a = .001*np.array([1,1,1]),return_f = True,tol_x = h,tol_f = h)
#    s_mu_p_ave = s_mu_p0_chi[0]
#    p_theory_p = s_mu_p_ave[1]/s_mu_p_ave[0]*(1-np.exp(-s_mu_p_ave[0]*tt)) +\
#    s_mu_p_ave[2]*np.exp(-s_mu_p_ave[0]*tt)
#    
#    
#    jj_ave_D = jj_sample[np.where(s_mu_D[:,0] <0.)]
#    p_ave_D = freqs[:,jjnuc0[jj_ave_D],jj_ave_D].mean(axis=1)
#    p_k = np.zeros(tt.shape[0]+1)
#    p_k[1:] = np.copy(p_ave_D)     
#    s_mu_D0 = h*np.array([1,1,1])
#    s_mu_D_L = vp.amoeba_vp(logLike_2nucs_exp_smuD,s_mu_D0,args=(p_k,t_k),\
#    a = .001*np.array([1,1,1]),return_f = True,tol_x = h,tol_f = h)
#    s_mu_D_ave = s_mu_D_L[0]
#    p_theory_D = s_mu_D_ave[1]/s_mu_D_ave[0]*(1-np.exp(-s_mu_D_ave[0]*tt))
#    
#    plt.figure(20,figsize=(20,5))
#    plt.clf()
#    plt.plot(tt,p_ave)
#    plt.plot(tt,p_ave_p)
#    plt.plot(tt,p_ave_D)
#    plt.plot(tt,p_theory)
#    plt.plot(tt,p_theory_p)
#    plt.plot(tt,p_theory_D)
#    plt.legend(('chi2','p0','D','chi2, theory','p0, theory','D, theory'),loc=0)
#    plt.xlabel('t')
#    plt.savefig(outdir_name + gen_region + '_'+nuc1+nuc2 + '_p_ave_negs.pdf')
#    plt.close(20)
#    
#
#    '''Trajectories'''
#    plt.figure(20,figsize=(20,15))
#    plt.clf()
#
#    plt.subplot(3,1,1)
#    for jnuc in jj_ave:
#        plt.plot(tt,freqs[:,jjnuc0[jnuc],jnuc])
#    plt.xlabel('t')
#    plt.ylabel('frequency')
#    plt.title('chi2')
#    
#    plt.subplot(3,1,2)
#    for jnuc in jj_ave_p:
#        plt.plot(tt,freqs[:,jjnuc0[jnuc],jnuc])
#    plt.xlabel('t')
#    plt.ylabel('frequency')
#    plt.title('p0')
#
#    plt.subplot(3,1,3)
#    for jnuc in jj_ave_D:
#        plt.plot(tt,freqs[:,jjnuc0[jnuc],jnuc])
#    plt.xlabel('t')
#    plt.ylabel('frequency')
#    plt.title('D')
#    
#    plt.savefig(outdir_name + gen_region + '_'+nuc1+nuc2 + '_traj_negs.pdf')
#    plt.close(20)
#    