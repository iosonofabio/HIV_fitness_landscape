# vim: fdm=indent
'''
author:     Vadim Puller
date:       04/02/16
content:    Compute results for the KL fit in Fig.2
'''
# Modules
from __future__ import division
import os
import sys
import argparse
import numpy as np
from scipy import linalg as LA
from scipy import optimize

from hivevo.patients import Patient
from hivevo.HIVreference import HIVreference



# Globals
h = 10**(-8)
cols_Fabio = ['b','c','g','y','r','m']
cols = ['b','g','r','c','m','y','k','b','g','r']
outdir_name = '../data/'


# Functions
def covariance(p_ka):
    '''Covariance matrix and correlation coefficient

    Input parameters:
    p_ka - nucleotide frequencies by site by time point

    Returns:
    time-covariance matrix
    '''
    pmean_k = p_ka.mean(axis=1); L = p_ka.shape[1]
    return (p_ka - np.tile(pmean_k,(L,1)).T).dot(p_ka.T - np.tile(pmean_k,(L,1)))/(L-1)


def fit_upper_multipat(xk_q_all, tk_all, LL=None):
    '''Fitting the quantile data with a linear law

    Input parameters:
    k_q_all - list of by-quantile mean frequencies for all patients
    tk_all - list of the time points for all patients
    LL - number of nucleotdies in the quantile for each patient

    Returns:
    slope (mutaion rate)
    '''

    def fit_upper_chi2(mu):
        chi2 = np.zeros(len(tk_all))
        for jpat, tk in enumerate(tk_all):
            chi2[jpat] = ((xk_q_all[jpat][-1,:] - mu*tk)**2).sum()
        return LL.dot(chi2)

    if LL is None:
        LL = np.ones(len(tk_all))
    res = optimize.minimize_scalar(fit_upper_chi2)
    return res.x


def KLfit_multipat_mu(Ckq_q_all, xk_q_all, tk_all, mu):
    '''
    Simultaneous KL divergence minimization for several quantiles

    Input parameters:
    Ckq_q_all - list of by-quantile covariance matrices for all patients
    xk_q_all - list of by-quantile mean frequencies for all patients
    tk_all - list of the time points for all patients
    mu - mutation rate

    Returns:
    fitness parameters for each quantile, mutation rate and noise parameter
    '''
    from scipy.optimize import minimize

    q = xk_q_all[0].shape[0]
    def akq_mat(s, dt_k):
        if s > h/np.min(dt_k):
            a0 =  2*s/(1-np.exp(-2*s*dt_k))
            a0[:-1] += 2*s*np.exp(-2*s*dt_k[1:])/(1-np.exp(-2*s*dt_k[1:]))
            a1 = - 2*s*np.exp(-s*dt_k[1:])/(1-np.exp(-2*s*dt_k[1:]))
        else:
            a0 =  1/dt_k
            a0[:-1] += (1.- s*dt_k[1:])**2/dt_k[1:]
            a1 = -(1.- s*dt_k[1:])/dt_k[1:]
        return (np.diag(a0) + np.diag(a1,1) + np.diag(a1,-1))

    def KL_multipat(sD):
        D0 = sD[-1]**2
        Like = np.zeros((len(tk_all),q))
        for jpat, tk in enumerate(tk_all):
            dtk = np.zeros(tk.shape[0]); dtk[0] = tk[0]; dtk[1:] = np.diff(tk)
            for jq in xrange(q):
                s = sD[jq]**2
                Akq = akq_mat(s,dtk)/D0
                if s > h/np.min(dtk):
                    b_k = mu*(1-np.exp(-s*tk))/s
                else:
                    b_k = mu*tk
                Like[jpat,jq] = -.5*np.log(LA.det(Akq)) + 0.5*(xk_q_all[jpat][jq,:]-b_k).dot(Akq).dot(xk_q_all[jpat][jq,:]-b_k)+\
                                                        + .5*np.trace(Ckq_q_all[jpat][jq,:,:].dot(Akq))
        if np.isnan(Like).any():
            print sD**2, Like
        return Like.sum()

    # we operate on the square root of the coefficients -> when squared positive, not bounds required
    sD0 = 1e-3*np.ones(q+1)
    step = 1e-4*np.ones(q+1)
    tol = h
    #sD = amoeba_vp(KL_multipat,sD0,args=(),a = step,tol_x = tol,tol_f = tol)
    sol = minimize(KL_multipat, sD0, method='Powell')
    if sol['success']:
        sD = sol['x']
        return sD**2
    else:
        print('Powell minimization failedm continue with Nelder-Mead')
        sol = minimize(KL_multipat, sD0, method='Nelder-Mead')
        if sol['success']:
            sD = sol['x']
            return sD**2
        else:
            import ipdb; ipdb.set_trace();


def patient_preprocessing(pat_name,Squant, div = False, outliers = True, xcut = 0.0,
                          xcut_up = 0.):
    '''Load patient data, remove outliers and return average frequencies by
    timepoint and covariances

    Input arguments:
    pat_name - patient name
    q - number of entropy categories

    Returns:
    xka_q - nucleotide frequencies by time-point, site, quantile
    tt - corresponding time values
    Smedians, Squant
    '''
    q=len(Squant)
    print pat_name
    PAT = Patient.load(pat_name)
    tt = PAT.times()
    map_to_ref = PAT.map_to_external_reference(gen_region)
    freqs = PAT.get_allele_frequency_trajectories(gen_region)[:,:,map_to_ref[:,2]]
#    if np.count_nonzero(freqs.mask) > 0:
#        print 'non-zero mask in ' + pat_name + ': removing ' +\
#        str(np.count_nonzero(freqs.mask.sum(axis=2).sum(axis=1)>0)) +\
#        ' time points out of ' + str(tt.shape[0])
#        tt = tt[np.where(freqs.mask.sum(axis=2).sum(axis=1) == 0)]
#        freqs = freqs[np.where(freqs.mask.sum(axis=2).sum(axis=1) == 0)[0],:,:]


    dt_k = np.array([tt[0]] + list(np.diff(tt)))
    jjnuc0 = np.argmax(freqs[0,:4,:],axis=0)
    anc_freq = freqs[:,jjnuc0,range(jjnuc0.shape[0])]
    xave = np.sum(dt_k * anc_freq.T, axis=1) /np.sum(dt_k * (~anc_freq.mask).T, axis=1)
    good_positions = (xave <= 1.-xcut)*(xave > xcut_up)*(jjnuc0==ref.get_consensus_indices_in_patient_region(map_to_ref))

    xka_q = []
    for jq in xrange(q):
        idx_ref = Squant[jq]['ind']
        idx_PAT = np.array([i for i,jref in enumerate(map_to_ref[:,0])
                            if (jref in idx_ref) and good_positions[i]])

        if div:
            x_ka = (1.- anc_freq[:,idx_PAT])*anc_freq[:,idx_PAT]
        else:
            x_ka = (1.- anc_freq[:,idx_PAT])
        xka_q.append(x_ka)

    # Remove outliers from the data
    if outliers:
        xka_q_new = []
        for jq, x_ka0 in enumerate(xka_q):
            nonout = np.all(x_ka0<0.5, axis=0)
            xka_q_new.append(xka0[:,nonout])
        xka_q = list(xka_q_new)

    return xka_q, tt

# Script
if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Fitness cost')
    parser.add_argument('--quantiles', type=int, default=6,
                        help="Number of quantiles")
    parser.add_argument('--subtype', type=str, default='any',
                        help="subtype to compare against")
    args = parser.parse_args()


    gen_region = 'genomewide' #'pol' #'gag' #'pol' #'gp41' #'gp120' #'vif' #'RRE'

    ref = HIVreference(subtype=args.subtype)
    tmp = ref.get_entropy_quantiles(args.quantiles)
    Squant = {}
    qi1=0
    for qi in range(len(tmp)): # prune empty quantiles.
        if len(tmp[qi]['ind']):
            Squant[qi1]=tmp[qi]
            qi1+=1

    q=len(Squant)
    Smedians = [np.median(ref.entropy[Squant[i]['ind']]) for i in range(q)]


    patient_names = ['p1','p2','p3','p5','p6','p8','p9','p11']
    # 'p4', 'p7' do not exist
    # 'p10' - weird messages from Patient class
    # p3, p8 - True mask for some time points

    # Input arguments
    tt_all = []; xk_q_all = []; Ckq_q_all = []
    Lq = []
    for jpat, pat_name in enumerate(patient_names):
        xka_q, tt = patient_preprocessing(pat_name,Squant, outliers=True)
        # Calculate mean and covariances
        xk_q = np.zeros((q,tt.shape[0]))
        Ckq_q = np.zeros((q,tt.shape[0],tt.shape[0]))
        for jq, x_ka in enumerate(xka_q):
            Lq.append(x_ka.shape[1])
            xk_q[jq,:] = x_ka.mean(axis=1)
            Ckq_q[jq,:,:] = np.ma.cov(x_ka)
            print('Covariance deviation:', np.sum((np.ma.cov(x_ka)-covariance(x_ka))**2))

        xk_q_all.append(xk_q)
        Ckq_q_all.append(Ckq_q)
        tt_all.append(tt)
    Lq = np.array(Lq, dtype=int)
    # Simultaneous fit for all patients given a mutation rate
    smuD_KL_q_multipat_mu = np.zeros(q+2)
    #smuD_KL_q_multipat_mu[q] = fit_upper_multipat(xk_q_all,tt_all)
    smuD_KL_q_multipat_mu[q] = 1.2e-5 # fixed mutation rate.
    ii_sD = [i for i in xrange(q+2) if i != q]
    smuD_KL_q_multipat_mu[ii_sD] = KLfit_multipat_mu(Ckq_q_all,xk_q_all,tt_all,smuD_KL_q_multipat_mu[q])

    # Saving fitness coefficients and mutation rates
    header = ['s' + str(jq+1) for jq in range(q)]
    header.extend(['mu','D'])
    if outdir_name is not None:
        np.savetxt(outdir_name + gen_region+'_smuD_KLmu_multi.txt',np.array([smuD_KL_q_multipat_mu]),header = '\t\t\t'.join(header))

        qbord = list(Squant[0]['range']) + [Squant[i]['range'][1] for i in xrange(1, len(Squant))]
        np.savetxt(outdir_name+gen_region+'_smuD_KL_quantiles.txt', qbord)
        np.savetxt(outdir_name+gen_region+'_smuD_KL_quant_medians.txt', Smedians)

    # Bootstrapping
    Nboot = 100
    smuD_multipat_boot_mu = np.zeros((Nboot,q+2))
    for jboot in xrange(Nboot):
        print 'bootstrap #', jboot
        pp = np.random.randint(0,len(patient_names),len(patient_names))
        smuD_multipat_boot_mu[jboot,q] = 1.2e-5 # fixed mutation rate
        smuD_multipat_boot_mu[jboot,ii_sD] = KLfit_multipat_mu([Ckq_q_all[p] for p in pp],\
        [xk_q_all[p] for p in pp],[tt_all[p] for p in pp],smuD_multipat_boot_mu[jboot,q])

    smuD_multipat_boot_mu_mean = smuD_multipat_boot_mu.mean(axis=0)
    smuD_multipat_boot_mu_sigma = np.sqrt((smuD_multipat_boot_mu**2).mean(axis=0) - smuD_multipat_boot_mu.mean(axis=0)**2)
    if outdir_name is not None:
        np.savetxt(outdir_name + gen_region+'_smuD_KLmu_multi_boot.txt',np.array([smuD_multipat_boot_mu_mean, smuD_multipat_boot_mu_sigma]),header = '\t\t\t'.join(header))
