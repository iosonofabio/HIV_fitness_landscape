# vim: fdm=indent
'''
author:     Richard Neher
date:       22/02/16
content:    Combine allele frequencies from all patients for strongly conserved sites
'''
# Modules
from __future__ import division, print_function

import os
import sys
import argparse
import cPickle
import gzip
from itertools import izip
from scipy.stats import spearmanr, scoreatpercentile, pearsonr
from random import sample
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns

from hivevo.patients import Patient
from hivevo.HIVreference import HIVreference
from hivevo.af_tools import divergence



# Globals
ERR_RATE = 2e-3
WEIGHT_CUTOFF = 500
SAMPLE_AGE_CUTOFF = 2 # years
af_cutoff = 1e-5
sns.set_style('darkgrid')
ls = {'gag':'-', 'pol':'--', 'nef':'-.'}
cols = sns.color_palette(n_colors=7)
fs = 16
regions = ['gag', 'pol', 'vif', 'vpr', 'vpu', 'env', 'nef']



# Functions
def load_mutation_rates():
    fn = '../data/mutation_rate.pickle'
    return pd.read_pickle(fn)


def running_average(obs, ws):
    '''Calculates a running average
    obs     --  observations
    ws      --  window size (number of points to average)
    '''
    try:
        tmp_vals = np.convolve(np.ones(ws, dtype=float)/ws, obs, mode='same')
        # fix the edges. using mode='same' assumes zeros outside the range
        if ws%2==0:
            tmp_vals[:ws//2]*=float(ws)/np.arange(ws//2,ws)
            if ws//2>1:
                tmp_vals[-ws//2+1:]*=float(ws)/np.arange(ws-1,ws//2,-1.0)
        else:
            tmp_vals[:ws//2]*=float(ws)/np.arange(ws//2+1,ws)
            tmp_vals[-ws//2:]*=float(ws)/np.arange(ws,ws//2,-1.0)
    except:
        import ipdb; ipdb.set_trace()
        tmp_vals = 0.5*np.ones_like(obs, dtype=float)
    return tmp_vals


def draw_genome(annotations,
                ax=None,
                rows=4,
                readingframe=True, fs=9,
                y1=0,
                height=1,
                pad=0.2):
    '''Draw genome boxes'''
    from matplotlib.patches import Rectangle
    from Bio.SeqFeature import CompoundLocation

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.set_ylim([-pad,rows*(height+pad)])
    anno_elements = []
    for name, feature in annotations.iteritems():
        if type(feature.location) is CompoundLocation:
            locs = feature.location.parts
        else:
            locs = [feature.location]
        for li,loc in enumerate(locs):
            x = [loc.nofuzzy_start, loc.nofuzzy_end]
            anno_elements.append({'name': name,
                                  'x1': x[0],
                                  'x2': x[1],
                                  'width': x[1] - x[0]})
            if name[0]=='V':
                anno_elements[-1]['ri']=3
            elif li:
                anno_elements[-1]['ri']=(anno_elements[-2]['ri'] + ((x[0] - anno_elements[-2]['x2'])%3))%3
            else:
                anno_elements[-1]['ri']=x[0]%3

    anno_elements.sort(key = lambda x:x['x1'])
    for ai, anno in enumerate(anno_elements):
        if readingframe:
            anno['y1'] = y1 + (height + pad) * anno['ri']
        else:
            anno['y1'] = y1 + (height + pad) * (ai%rows)
        anno['y2'] = anno['y1'] + height
        anno['height'] = height

    for anno in anno_elements:
        r = Rectangle((anno['x1'], anno['y1']),
                      anno['width'],
                      anno['height'],
                      facecolor=[0.8] * 3,
                      edgecolor='k',
                      label=anno['name'])

        xt = anno['x1'] + 0.5 * anno['width']
        yt = anno['y1'] + 0.2 * height + height * (anno['width']<500)
        anno['x_text'] = xt
        anno['y_text'] = yt

        ax.add_patch(r)
        ax.text(xt, yt,
                anno['name'],
                color='k',
                fontsize=fs,
                ha='center')

    return pd.DataFrame(anno_elements)


def patient_bootstrap(afs):
    '''
    resample allele frequencies of patients to produce pseudo replicates of equal size
    '''
    patients = afs.keys()
    tmp_sample = np.random.randint(len(patients), size=len(patients))
    return [afs[patients[ii]] for ii in tmp_sample]


def patient_partition(afs):
    '''
    take a set of allele frequencies for different patients and split
    it into two random partitions of equal size (like bootstraping but without resampling)
    '''
    patients = afs.keys()
    tmp_sample = sample(patients, len(patients)//2)
    remainder = set(patients).difference(tmp_sample)
    return [afs[pat] for pat in tmp_sample], [afs[pat] for pat in remainder]


def af_average(afs):
    '''Average weighted allele frequency estimates'''
    tmp_afs = np.sum(afs, axis=0)
    tmp_afs[:,tmp_afs.sum(axis=0)==0] = np.nan
    tmp_afs = tmp_afs/(np.sum(tmp_afs, axis=0)+1e-6)
    return tmp_afs


def collect_weighted_afs(region, patients, reference, cov_min=1000, max_div=0.5):
    '''
    produce weighted averages of allele frequencies for all late samples in each patients
    restrict to sites that don't sweep and have limited diversity as specified by max_div
    '''
    good_pos_in_reference = reference.get_ungapped(threshold = 0.05)
    combined_af_by_pat = {}
    syn_nonsyn_by_pat={}
    syn_nonsyn_by_pat_unconstrained={}
    region_start = int(reference.annotation[region].location.start)
    for pi, p in enumerate(patients):
        pcode= p.name
        print("averaging ",pcode," region ",region)
        try:
            pcode= p.name
            combined_af_by_pat[pcode] = np.zeros((6, len(reference.annotation[region])))
            print(pcode, p.Subtype)
            aft = p.get_allele_frequency_trajectories(region, cov_min=cov_min, error_rate = ERR_RATE, type='nuc')

            # get patient to subtype map
            patient_to_subtype = p.map_to_external_reference(region, refname=reference.refname)
            consensus = reference.get_consensus_indices_in_patient_region(patient_to_subtype)
            ref_ungapped = good_pos_in_reference[patient_to_subtype[:,0]]

            ancestral = p.get_initial_indices(region)[patient_to_subtype[:,2]]
            rare = ((aft[:,:4,:]**2).sum(axis=1).min(axis=0)>max_div)[patient_to_subtype[:,2]]
            final = aft[-1].argmax(axis=0)[patient_to_subtype[:,2]]

            syn_nonsyn_by_pat[pcode] = np.zeros(len(reference.annotation[region]), dtype=int)
            syn_nonsyn_by_pat[pcode][patient_to_subtype[:,0]-patient_to_subtype[0][0]]+=\
                (p.get_syn_mutations(region, mask_constrained=True).sum(axis=0)>1)[patient_to_subtype[:,2]]
            syn_nonsyn_by_pat_unconstrained[pcode] = np.zeros(len(reference.annotation[region]), dtype=int)
            syn_nonsyn_by_pat_unconstrained[pcode][patient_to_subtype[:,0]-patient_to_subtype[0][0]]+=\
                (p.get_syn_mutations(region, mask_constrained=False).sum(axis=0)>1)[patient_to_subtype[:,2]]
            for af, ysi, depth in izip(aft, p.ysi, p.n_templates_dilutions):
                if ysi<SAMPLE_AGE_CUTOFF:
                    continue
                pat_af = af[:,patient_to_subtype[:,2]]
                patient_consensus = pat_af.argmax(axis=0)
                ind = ref_ungapped&rare&(patient_consensus==consensus)&(ancestral==consensus)&(final==consensus)
                w = depth/(1.0+depth/WEIGHT_CUTOFF)
                combined_af_by_pat[pcode][:,patient_to_subtype[ind,0]-region_start] \
                            += w*pat_af[:,ind]
        except:
            import ipdb; ipdb.set_trace()
    return combined_af_by_pat, syn_nonsyn_by_pat, syn_nonsyn_by_pat_unconstrained


def process_average_allele_frequencies(data, regions,
                                       nbootstraps=0,
                                       bootstrap_type='bootstrap',
                                       synnonsyn=False,
                                       nstates=4):
    '''
    calculate the entropies, minor frequencies etc from the individual patient averages
    boot strap on demand
    '''
    combined_af={}
    combined_entropy={}
    minor_af={}

    # boot straps
    minor_af_bs={}
    combined_entropy_bs={}
    for region in regions:
        combined_af[region] = af_average(data['af_by_pat'][region].values())
        combined_entropy[region] = (-np.log2(combined_af[region]+1e-10)*combined_af[region]).sum(axis=0)
        valid_states = combined_af[region][:nstates,:]
        minor_af[region] = (valid_states.sum(axis=0) - valid_states.max(axis=0))/(1e-6+valid_states.sum(axis=0))
        #ind = combined_af[region][:nstates,:].sum(axis=0)<0.5
        #minor_af[region][ind]=np.nan
        #combined_entropy[region][ind]=np.nan
        if nbootstraps:
            minor_af_bs[region]=[]
            combined_entropy_bs[region]=[]
            if bootstrap_type=='bootstrap':
                for ii in xrange(nbootstraps):
                    tmp_af = af_average(patient_bootstrap(data['af_by_pat'][region]))
                    combined_entropy_bs[region].append((-np.log2(tmp_af+1e-10)*tmp_af).sum(axis=0))
                    minor_af_bs[region].append((tmp_af[:nstates,:].sum(axis=0) - tmp_af.max(axis=0))/(tmp_af[:nstates,:].sum(axis=0)+1e-6))
                    #ind = tmp_af[:nstates,:].sum(axis=0)<0.5
                    #minor_af_bs[region][-1][ind]=np.nan
                    #combined_entropy_bs[region][-1][ind]=np.nan
            elif bootstrap_type=='partition':
                for ii in xrange(nbootstraps//2):
                    for a in patient_partition(data['af_by_pat'][region]):
                        tmp_af = af_average(a)
                        combined_entropy_part[region].append((-np.log2(tmp_af+1e-10)*tmp_af).sum(axis=0))
                        minor_af_part[region].append((tmp_af[:nstates,:].sum(axis=0) - tmp_af.max(axis=0))/(tmp_af[:nstates,:].sum(axis=0)+1e-6))

    output = {'combined_af': combined_af,
              'combined_entropy': combined_entropy,
              'minor_af': minor_af,
             }

    if synnonsyn:
        # if a site is syn in greater than half the number of patients, call as syn (factor 2 on LHS)
        synnonsyn = {region: 2*np.array([x for x in data['syn_by_pat'][region].values()]).sum(axis=0)>len(data['syn_by_pat'][region])
                     for region in regions}
        # repeat for syn score that does not account for overlapping reading frames
        synnonsyn_unconstrained = {region: 2*np.array([x for x in data['syn_by_pat_uc'][region].values()]).sum(axis=0)>len(data['syn_by_pat_uc'][region])
                     for region in regions}
        output['synnonsyn'] = synnonsyn
        output['synnonsyn_unconstrained'] = synnonsyn_unconstrained

    if nbootstraps:
        output['combined_entropy_bs'] = combined_entropy_bs
        output['minor_af_bs'] = minor_af_bs

    return output


def entropy_scatter(region, within_entropy, synnonsyn, reference, fname = None, running_avg=True):
    '''
    scatter plot of cross-sectional entropy vs entropy of averaged intrapatient frequencies
    '''
    xsS = np.array([reference.entropy[ii] for ii in reference.annotation[region]])
    ind = (xsS>=0.000)&(~np.isnan(within_entropy[region]))
    print(region)
    print("Pearson:", pearsonr(within_entropy[region][ind], xsS[ind]))
    rho, pval = spearmanr(within_entropy[region][ind], xsS[ind])
    print("Spearman:", rho, pval)

    npoints=20
    thres_xs = [0.0, 0.1, 10.0]
    thres_xs = zip(thres_xs[:-1],thres_xs[1:])
    nthres_xs=len(thres_xs)
    thres_in = [0.0, 0.0001, 10.0]
    thres_in = zip(thres_in[:-1],thres_in[1:])
    nthres_in=len(thres_in)
    enrichment = np.zeros((2,nthres_in, nthres_xs), dtype=int)
    plt.figure(figsize = (7,6))
    for ni, syn_ind, label_str in ((0, ~synnonsyn[region], 'nonsynymous'), (2,synnonsyn[region], 'synonymous')):
        tmp_ind = ind&syn_ind
        plt.scatter(within_entropy[region][tmp_ind]+.00003, xsS[tmp_ind]+.005, c=cols[ni], label=label_str, s=30)
        if running_avg:
            A = np.array(sorted(zip(within_entropy[region][tmp_ind]+.00003, xsS[tmp_ind]+0.005), key=lambda x:x[0]))
            plt.plot(np.exp(np.convolve(np.log(A[:,0]), np.ones(npoints, dtype=float)/npoints, mode='valid')),
                        np.exp(np.convolve(np.log(A[:,1]), np.ones(npoints, dtype=float)/npoints, mode='valid')),
                        c=cols[ni], lw=3)

            for ti1,(tl1, tu1) in enumerate(thres_in):
                for ti2,(tl2, tu2) in enumerate(thres_xs):
                    enrichment[ni>0, ti1, ti2] = np.sum((A[:,0]>=tl1)&(A[:,0]<tu1)&(A[:,1]>=tl2)&(A[:,1]<tu2))

    from scipy.stats import fisher_exact
    print(enrichment, fisher_exact(enrichment[:,:,1]))

    plt.ylabel('cross-sectional entropy', fontsize=fs)
    plt.xlabel('pooled within patient entropy', fontsize=fs)
    plt.text(0.00002, 1.3, r"Combined Spearman's $\rho="+str(round(rho,2))+"$", fontsize=fs)
    plt.legend(loc=4, fontsize=fs*0.8)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim([0.001, 2])
    plt.xlim([0.00001, .3])
    plt.tick_params(labelsize=fs*0.8)
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)
    return enrichment


def fraction_diverse(region, minor_af, synnonsyn, fname=None):
    '''Cumulative figures of the frequency distributions'''
    plt.figure()
    for ni, ind, label_str in ((0, ~synnonsyn[region], 'nonsynonymous'), (2,synnonsyn[region], 'synonymous')):
        tmp_ind = ind&(~np.isnan(minor_af[region]))
        plt.plot(sorted(minor_af[region][tmp_ind]+0.00001), np.linspace(0,1,tmp_ind.sum()),
                label=label_str+' n='+str(np.sum(tmp_ind)))
    plt.xscale('log')
    plt.yscale('linear')
    plt.legend(loc=2, fontsize=fs*0.8)
    plt.xlabel('minor frequency X', fontsize=fs)
    plt.ylabel(r'fraction less diverse than X', fontsize=fs)
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)

def entropy_correlation_vs_npat(region, data, synnonsyn, reference):
    '''
    evaluate entropy within/cross-sectional correlation for subsets of patients
    of different size. returns a dictionary with rank correlation coefficients
    '''
    from scipy.special import binom
    from random import sample
    from collections import defaultdict
    pats = data['af_by_pat'][region].keys()
    N = len(pats)
    within_cross_correlation = defaultdict(list)
    xsS = np.array([reference.entropy[ii] for ii in reference.annotation[region]])
    for n in range(1,1+N):
        for ii in range(int(min(2*binom(N,n),20))):
            subset = [data['af_by_pat'][region][x] for x in sample(pats, n)]
            tmp_af = af_average(subset)
            withS = -(np.log2(tmp_af+1e-10)*tmp_af).sum(axis=0)
            ind = (xsS>=0.000)&(~np.isnan(withS))
            within_cross_correlation[n].append(spearmanr(xsS[ind], withS[ind])[0])

    return within_cross_correlation

def SvsNpat(data, synnonsyn, reference, figname=None):
    '''
    calculate cross-sectional and within patient entropy correlations
    for many subsets of patients and plot the average rank correlation
    against the number of patients used in the within patient average
    '''
    for region in ['gag', 'pol', 'vif', 'nef']:
        xsS_within_corr = entropy_correlation_vs_npat(region, data, synnonsyn, reference)
        npats = sorted(xsS_within_corr.keys())
        avg_corr = [np.mean(xsS_within_corr[i]) for i in npats]
        std_corr = [np.std(xsS_within_corr[i]) for i in npats]
        plt.errorbar([1.0/i for i in npats], y=avg_corr,yerr=std_corr, label=region)
    plt.legend()
    plt.ylabel('within/cross-sectional entropy correlation', fontsize=fs)
    plt.xlabel('1/number of patients', fontsize=fs)
    plt.xlim(0,1.1)
    plt.tight_layout()
    if figname is not None:
        for ext in ['png', 'svg', 'pdf']:
            plt.savefig(figname+'.'+ext)



def selcoeff_distribution(regions, minor_af, synnonsyn, synnonsyn_uc, mut_rates, fname=None, ref=None):
    '''
    produce figure of distribution of selection coefficients separately for
    synonymous and nonsynonymous sites.
    '''
    from util import add_panel_label

    if ref is not None:
        if not hasattr(ref, 'fitness_cost'):
            ref.fitness_cost = np.zeros_like(ref.entropy)
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(10,6))
    #plt.title(region+' selection coefficients')
    if type(regions)==str:
        regions = [regions]
    for ni,ax,label_str in ((0, axs[0], 'synonymous'),
                            (1, axs[1], 'syn-overlaps'),
                            (2, axs[2], 'nonsyn')):
        slist = []
        for region in regions:
            if label_str=='synonymous':
                ind = synnonsyn[region]
            elif label_str=='syn-overlaps':
                ind = synnonsyn_uc[region]&(~synnonsyn[region])
            else:
                ind = ~synnonsyn_uc[region]
            ind = ind&(~np.isnan(minor_af[region]))
            slist.extend(mut_rates[region][ind]/(minor_af[region][ind]+af_cutoff))
        s = np.array(slist)
        s[s>=0.1] = 0.1
        s[s<=0.001] = 0.001
        if ref is not None:
            bg = ref.annotation[region].location.start
            ed = ref.annotation[region].location.end
            ref.fitness_cost[bg:ed][ind] = s
        if len(s):
            ax.hist(s, color=cols[ni],
                 weights=np.ones(len(s), dtype=float)/len(s), bins=np.logspace(-3,-1,11), label=label_str+', n='+str(len(s)))
        ax.set_xscale('log')
        ax.tick_params(labelsize=fs*0.8)
        ax.text(0.1, 0.8, 'position: '+str(ni))
        if ni==0:
            ax.set_ylabel('fraction of sites', fontsize=fs)
            ax.set_yscale('linear')
        ax.set_xlabel('selection coefficient', fontsize=fs)
        ax.set_xticks([0.001, 0.01, 0.1])
        ax.set_xticklabels([r'$<10^{-3}$', r'$10^{-2}$', r'$>10^{-1}$'])
        ax.legend(loc=2, fontsize=fs*0.8)

        add_panel_label(ax, ['A', 'B', 'C'][ni],
                        x_offset=-0.2 - 0.1 * (ni == 0))

    plt.tight_layout()
    if fname is not None:
        for ext in ['png', 'svg', 'pdf']:
            plt.savefig(fname+'.'+ext)


def selcoeff_confidence(region, data, fname=None):
    '''
    bootstrap the selection coefficients and make distributions of the bootstrapped
    values for subsets of sites with a defined median. this should give an impression
    of how variable the estimates are. three such distributions are combined in one
    figure
    '''
    from util import add_panel_label

    # generate boo strap estimates of minor SNP frequences
    av = process_average_allele_frequencies(data, [region],
                    nbootstraps=100, bootstrap_type='bootstrap')
    combined_af = av['combined_af']
    combined_entropy = av['combined_entropy']
    minor_af = av['minor_af']
    combined_entropy_bs = av['combined_entropy_bs']
    minor_af_bs = av['minor_af_bs']

    # convert minor_af to 100x(length of gene) array of minor SNPs
    minor_af_array=np.array(minor_af_bs[region])
    qtiles = np.vstack([scoreatpercentile(minor_af_array, x, axis=0) for x in [25, 50, 75]])
    # calculate selection coefficient quantiles corresponding to SNP_freq quantiles
    scb = (data['mut_rate'][region]/(af_cutoff+qtiles)).T
    sel_coeff_array = (data['mut_rate'][region]/(af_cutoff+minor_af_array))
    sel_coeff_array[sel_coeff_array<0.001]=0.001
    sel_coeff_array[sel_coeff_array>0.1]=0.1
    which_quantile = np.zeros(minor_af_array.shape[1], dtype=int)
    thres = [20,40,60]
    for i,ql in enumerate(thres):
        # take sites if slice [ql,ql+2]
        sl,su=scoreatpercentile(scb[:,1], ql), scoreatpercentile(scb[:,1], ql+2)
        which_quantile[(scb[:,1]>=sl)&(scb[:,1]<su)]=i+1

    scb[scb>0.1]=0.1
    scb[scb<0.001]=0.001
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    for i in range(1,len(thres)+1):
        ind = which_quantile==i
        npoints = ind.sum()*sel_coeff_array.shape[0]
        ax.plot(np.median(scb[ind,1])*np.ones(2), [0,0.5], c=cols[i+3], lw=4)
        ax.hist(sel_coeff_array[:,ind].flatten(), weights =np.ones(npoints,dtype=float)/npoints,
                bins=np.logspace(-3,-1,21), alpha=0.7, color=cols[i+3])
    ax.set_xscale('log')
    ax.set_xlabel('selection coefficient', fontsize=fs)
    ax.set_ylabel('uncertainty distribution', fontsize=fs)
    plt.tick_params(labelsize=fs*0.8)

    plt.tight_layout()
    region_panels = {'gag': 'D', 'pol': 'A', 'env': 'B', 'nef': 'C', 'vif': 'D',
                     'vpu': 'E', 'vpr': 'F'}
    add_panel_label(ax, region_panels.get(region, 'D'), x_offset=-0.1)

    if fname is not None:
        for ext in ['png', 'svg', 'pdf']:
            plt.savefig(fname+'.'+ext)


def selcoeff_vs_entropy(regions,  minor_af, synnonsyn, mut_rate, reference,
                        figname=None,
                        dataname=None,
                        smoothing='harmonic'):
    fig, ax = plt.subplots()
    npoints=20
    avg_sel_coeff = {}
    #plt.title(region+' selection coefficients')
    if type(regions)==str:
        regions=[regions]
    for ni,label_str in ((0,'synonymous'), (1,'nonsynonymous'), (2,'all')):
        s=[]
        entropy = []
        for region in regions:
            xsS = np.array([reference.entropy[ii] for ii in reference.annotation[region]])
            ind = synnonsyn[region] if label_str=='synonymous' else ~synnonsyn[region]
            if label_str == 'all': ind = xsS>=0
            s.append(mut_rate[region][ind]/(minor_af[region][ind]+af_cutoff))
            entropy.append(xsS[ind])

        s = np.concatenate(s)
        entropy = np.concatenate(entropy)
        if label_str!='all':
            ax.scatter(entropy, s, c=cols[ni])

        A = np.array(sorted(zip(entropy+0.001, s), key=lambda x:x[0]))
        A = A[~np.isnan(A[:,1]),:]
        #ax.plot(np.exp(np.convolve(np.log(A[:,0]), 1.0*np.ones(npoints)/npoints, mode='valid')),
        #            np.exp(np.convolve(np.log(A[:,1]), 1.0*np.ones(npoints)/npoints, mode='valid')), c=cols[ni], label=label_str, lw=3)
        #ax.plot(np.convolve(A[:,0], 1.0*np.ones(npoints)/npoints, mode='valid'),
        #        np.convolve(A[:,1], 1.0*np.ones(npoints)/npoints, mode='valid'), c=cols[ni], label=label_str, lw=3)

        entropy_thresholds =  np.array(np.linspace(0,A.shape[0],8), int)
        entropy_boundaries = zip(entropy_thresholds[:-1], entropy_thresholds[1:])
        if smoothing=='harmonic':
            tmp_mean_inv= [(np.median(A[li:ui,0]), np.mean(1.0/A[li:ui,1], axis=0))
                            for li,ui in entropy_boundaries]
            avg_sel_coeff[label_str] = np.array([(xsSmed, 1.0/avg_inv)
                                                for xsSmed, avg_inv in tmp_mean_inv])
            avg_sel_coeff[label_str+'_std'] = np.array([(np.median(A[li:ui,0]),
                                            (avg_inv**-2)*np.std(1.0/A[li:ui,1], axis=0)/np.sqrt(ui-li))
                                            for (li,ui), (_a,avg_inv) in zip(entropy_boundaries,tmp_mean_inv)])
        elif smoothing=='median':
            avg_sel_coeff[label_str] = np.array([np.median(A[li:ui,:], axis=0) for li,ui in entropy_boundaries])
            avg_sel_coeff[label_str+'_std'] = np.array([(np.median(A[li:ui,0]), np.std(A[li:ui,1], axis=0))
                                                        for li,ui in entropy_boundaries])
        elif smoothing=='geometric':
            avg_sel_coeff[label_str] = np.array([(np.median(A[li:ui,0]), np.exp(np.mean(np.log(A[li:ui,1]), axis=0)))
                                                 for li,ui in entropy_boundaries])
            avg_sel_coeff[label_str+'_std'] = np.array([(np.median(A[li:ui,0]), np.exp(np.std(np.log(A[li:ui,1], axis=0))))
                                                        for li,ui in entropy_boundaries])

        ax.plot(avg_sel_coeff[label_str][:,0], avg_sel_coeff[label_str][:,1], lw=3)

    ax.legend()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('Fitness cost')
    ax.set_xlabel('Variability in group M [bits]')

    plt.tight_layout()
    if figname is not None:
        plt.savefig(figname)

    if dataname is not None:
        with open(dataname, 'w') as ofile:
            cPickle.dump(avg_sel_coeff, ofile)

    return avg_sel_coeff


def plot_selection_coefficients_along_genome(regions, data, minor_af, synnonsyn, reference, ws=30):
    '''Plot the fitness costs along the genome'''
    from util import add_panel_label

    all_sel_coeff = []

    # Fitness costs along the genome
    fig, axs = plt.subplots(2, 1, sharex=True,
                            gridspec_kw={'height_ratios':[6, 1]})

    for ni,label_str in ((1,'nonsynonymous'), (0,'synonymous')):
        for ri, region in enumerate(regions):
            ind = synnonsyn[region] if label_str=='synonymous' else ~synnonsyn[region]
            ind = ind&(~np.isnan(minor_af[region]))
            #axs[0].plot([x for x in reference.annotation[region] if x%3==0], 1.0/np.convolve(np.ones(ws, dtype=float)/ws, 1.0/sc[region], mode='same'), c=cols[ri])
            sc = (data['mut_rate'][region]/(af_cutoff+minor_af[region]))
            sc[sc>0.1] = 0.1
            sc[sc<0.001] = 0.001
            axs[0].plot(running_average(np.array(list(reference.annotation[region]))[ind], ws),
                        np.exp(running_average(np.log(sc[ind]), ws)),
                        c=cols[ri%len(cols)],
                        ls='--' if label_str=='synonymous' else '-',
                       label=label_str if region=='gag' else None)
            if ni and region not in ['vpr', 'vpu']:
                all_sel_coeff.extend([(region, pos, np.log10(sc[pos]), synnonsyn[region][pos]) for pos in range(len(sc))])

    axs[0].legend(loc=1, fontsize=fs*0.8)
    axs[0].set_yscale('log')
    axs[0].set_ylabel('selection coefficient [1/day]', fontsize=fs)
    axs[0].set_ylim(0.002, 0.25)
    axs[0].tick_params(labelsize=fs*0.8)

    # The genome annotations
    regs = ['p17', 'p6', 'p7', 'p24',
            'PR', 'RT', 'IN', 'p15',
            'nef',
            'gp120', 'gp41',
            'vif', 'vpu', 'vpr', 'rev', 'tat',
            'V1', 'V2', 'V3', 'V5']
    annotations = {k: val for k, val in reference.annotation.iteritems() if k in regs}
    annotations = draw_genome(annotations, axs[1])
    axs[1].set_axis_off()
    feas = ['p17', 'p24', 'PR', 'RT', 'p15', 'IN', 'vif', 'gp120', 'gp41', 'nef']
    vlines = np.unique(annotations.loc[annotations['name'].isin(feas), ['x1', 'x2']])
    for xtmp in vlines:
        axs[0].axvline(xtmp, lw=1, color='0.8')

    plt.tight_layout()
    add_panel_label(axs[0], 'A', x_offset=-0.1)
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig('../figures/figure_4A_st_' + reference.subtype + '.'+ext)


    # Violin plots of the fitness cost distributions for syn and nonsyn
    all_sel_coeff = pd.DataFrame(data=all_sel_coeff, columns=['gene', 'position', 'selection', 'synonymous'])
    all_sel_coeff.loc[all_sel_coeff['synonymous'] == True, 'synonymous'] = 'synonymous'
    all_sel_coeff.loc[all_sel_coeff['synonymous'] == False, 'synonymous'] = 'nonsynonymous'
    fig = plt.figure()
    ax = sns.violinplot(x='gene', y='selection', hue='synonymous', data=all_sel_coeff,
                       inner='quartile', split=True, cut=0, scale='area')
    ax.set_yticks([-3,-2,-1])
    ax.set_yticklabels([r'$10^{'+str(i)+'}$' for i in [-3,-2,-1]])
    ax.tick_params(labelsize=0.8*fs)
    ax.set_ylabel('selection coefficient [1/day]', fontsize=fs)
    ax.set_xlabel('')
    ax.set_ylim(-3, -0.5)
    ax.legend(loc=1, fontsize=fs, title=None)

    plt.tight_layout()
    add_panel_label(ax, 'B', x_offset=-0.1)
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig('../figures/figure_4B_st_' + reference.subtype +'.'+ext)


def enrichment_analysis(regions, combined_entropy, synnonsyn, reference, minor_af):
    '''Enrichment of nonsynonymous mutations at globally variable but intrapatient conserved sites'''
    from scipy.stats import fisher_exact
    E = np.zeros((2,2,2))
    for region in regions:
        E += entropy_scatter(region, combined_entropy, synnonsyn, reference,
                             '../figures/'+region+'_entropy_scatter.png')
        fraction_diverse(region, minor_af, synnonsyn,
                         '../figures/'+region+'_minor_allele_frequency.pdf')
    print('NonSyn enrichment among variable sites with low within diversity',fisher_exact(E[:,:,1]))



def export_selection_coefficients(data, synnonsyn, subtype):
    from scipy.stats import scoreatpercentile
    def sel_out(s):
        if s<0.001:
            return '<0.001'
        elif s>0.1:
            return '>0.1'
        else:
            return s

    for region in data['af_by_pat']:
        av = process_average_allele_frequencies(data, [region],
                        nbootstraps=100, bootstrap_type='bootstrap')
        combined_af = av['combined_af']
        combined_entropy = av['combined_entropy']
        minor_af = av['minor_af']
        combined_entropy_bs = av['combined_entropy_bs']
        minor_af_bs = av['minor_af_bs']

        minor_af_array=np.array(minor_af_bs[region])
        qtiles = np.vstack([scoreatpercentile(minor_af_array, x, axis=0) for x in [25, 50, 75]])
        scb = (data['mut_rate'][region]/(af_cutoff+qtiles))

        with open('../data/nuc_'+region+'_selection_coeffcients_'+ subtype +'.tsv','w') as selfile:
            selfile.write('### selection coefficients in '+region+'\n')
            selfile.write('# position\tlower quartile\tmedian\tupper quartile \t syn \n')

            for pos in xrange(scb.shape[1]):
                selfile.write('\t'.join(map(str,[pos+1]+[sel_out(scb[qi][pos]) for qi in range(scb.shape[0])]
                            +[int(synnonsyn[region][pos])]))+'\n')


def collect_data(patient_codes, regions, reference):
    '''
    loop over regions and produce a dictionary that contains the frequencies,
    syn/nonsyn designations and mutation rates
    '''
    cov_min=1000
    combined_af_by_pat={}
    syn_nonsyn_by_pat={}
    syn_nonsyn_by_pat_unconstrained={}
    consensus_mutation_rate={}
    mutation_rates = load_mutation_rates()['mu']
    total_muts = {nuc: sum([x for mut, x in mutation_rates.iteritems() if mut[0]==nuc]) for nuc in 'ACGT'}

    patients = []
    for pcode in patient_codes:
        p = Patient.load(pcode)
        patients.append(p)
    for region in regions:
        combined_af_by_pat[region], syn_nonsyn_by_pat[region], syn_nonsyn_by_pat_unconstrained[region] \
            = collect_weighted_afs(region, patients, reference)
        consensus_mutation_rate[region] = np.array([total_muts[nuc] if nuc!='-' else np.nan for nuc in
                            reference.annotation[region].extract("".join(reference.consensus))])

    return {'af_by_pat': combined_af_by_pat,
            'mut_rate': consensus_mutation_rate,
            'syn_by_pat': syn_nonsyn_by_pat,
            'syn_by_pat_uc': syn_nonsyn_by_pat_unconstrained}



# Script
if __name__=="__main__":

    parser = argparse.ArgumentParser(description='nucleotide allele frequencies, saturation levels, and fitness costs')
    parser.add_argument('--regenerate', action='store_true',
                        help="regenerate data")
    parser.add_argument('--subtype', choices=['B', 'any'], default='B',
                        help='subtype to compare against')
    args = parser.parse_args()

    # NOTE: HXB2 alignment has way more sequences resulting in better correlations
    reference = HIVreference(refname='HXB2', subtype=args.subtype)

    # Intermediate data are saved to file for faster access later on
    fn = '../data/avg_nucleotide_allele_frequency.pickle.gz'
    if not os.path.isfile(fn) or args.regenerate:
        if args.subtype=='B':
            patient_codes = ['p2','p3', 'p5', 'p8', 'p9','p10', 'p11'] # subtype B only
            #patient_codes = ['p2','p3', 'p4', 'p5', 'p7', 'p8', 'p9','p10', 'p11'] # subtype B only
        else:
            patient_codes = ['p1', 'p2','p3','p5','p6', 'p8', 'p9','p10', 'p11'] # all subtypes, no p4/7
            #patient_codes = ['p1', 'p2','p3','p4', 'p5','p6','p7', 'p8', 'p9','p10', 'p11'] # patients

        data = collect_data(patient_codes, regions, reference)
        try:
            with gzip.open(fn, 'w') as ofile:
                cPickle.dump(data, ofile)
            print('Data saved to file:', os.path.abspath(fn))
        except IOError:
            print('Could not save data to file:', os.path.abspath(fn))

    else:
        with gzip.open(fn) as ifile:
            data = cPickle.load(ifile)

    # Check whether all regions are present
    if not all([region in data['mut_rate'] for region in regions]):
        print("data loading failed or data doesn't match specified regions:",
              regions, ' got:',data['mut_rate'].keys())

    # Average, annotate, and process allele frequencies
    av = process_average_allele_frequencies(data, regions, nbootstraps=0, synnonsyn=True)
    combined_af = av['combined_af']
    combined_entropy = av['combined_entropy']
    minor_af = av['minor_af']
    synnonsyn = av['synnonsyn']
    synnonsyn_unconstrained = av['synnonsyn_unconstrained']

    # Enrichment of nonsynonymous mutations at globally variable but intrapatient conserved sites
    enrichment_analysis(regions, combined_entropy, synnonsyn, reference, minor_af)

    # Prepare data for Figure 2 (see figure_2.py for the plot)
    selcoeff_vs_entropy(regions,  minor_af, synnonsyn, data['mut_rate'],
                        reference,
                        figname='../figures/'+region+'_sel_coeff_scatter_st_'+args.subtype+'.png',
                        dataname='../data/combined_af_avg_selection_coeff_st_'+args.subtype+'.pkl',
                        smoothing='harmonic')

    # Figure 3
    for region in regions:
        selcoeff_distribution(region, minor_af, synnonsyn, synnonsyn_unconstrained,
                               data['mut_rate'],
                              '../figures/'+region+'_sel_coeff_st_'+args.subtype, ref=reference)
        selcoeff_confidence(region, data,
                            '../figures/'+region+'_sel_coeff_confidence_st_'+args.subtype)

    selcoeff_distribution(['gag', 'pol', 'vif', 'vpu', 'vpr', 'nef'], minor_af, synnonsyn, synnonsyn_unconstrained,
                          data['mut_rate'],
                          '../figures/figure_3ABC_st_'+args.subtype)

    # Figure 4
    plot_selection_coefficients_along_genome(regions, data, minor_af, synnonsyn_unconstrained, reference)

    # export selection coefficients to file as supplementary info
    export_selection_coefficients(data, synnonsyn, args.subtype)
