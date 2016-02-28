# vim: fdm=indent
'''
author:     Richard Neher
date:       22/02/16
content:    Combine allele frequencies from all patients for strongly conserved sites
'''
# Modules
from __future__ import division, print_function
import sys, argparse, cPickle, os, gzip
sys.path.append('/ebio/ag-neher/share/users/rneher/HIVEVO_access/')
import numpy as np
from itertools import izip
from hivevo.patients import Patient
from hivevo.HIVreference import HIVreference
from hivevo.af_tools import divergence
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns
from scipy.stats import spearmanr, scoreatpercentile, pearsonr
from random import sample



# Globals
sns.set_style('darkgrid')
ls = {'gag':'-', 'pol':'--', 'nef':'-.'}
cols = sns.color_palette()
fs=16
plt.ion()



# Functions
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
    '''
    average weighted allele frequency estimates.
    '''
    tmp_afs = np.sum(afs, axis=0)
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
    for pi, p in enumerate(patients):
        pcode= p.name
        combined_af_by_pat[pcode] = np.zeros((6, len(reference.annotation[region])))
        print(pcode, p.Subtype)
        aft = p.get_allele_frequency_trajectories(region, cov_min=cov_min, error_rate = 2e-3, type='nuc')

        # get patient to subtype map
        patient_to_subtype = p.map_to_external_reference(region, refname = reference.refname)
        consensus = reference.get_consensus_indices_in_patient_region(patient_to_subtype)
        ref_ungapped = good_pos_in_reference[patient_to_subtype[:,0]]

        ancestral = p.get_initial_indices(region)[patient_to_subtype[:,2]]
        rare = ((aft[:,:4,:]**2).sum(axis=1).min(axis=0)>max_div)[patient_to_subtype[:,2]]
        final = aft[-1].argmax(axis=0)[patient_to_subtype[:,2]]

        syn_nonsyn_by_pat[pcode] = np.zeros(len(reference.annotation[region]), dtype=int)
        syn_nonsyn_by_pat[pcode][patient_to_subtype[:,0]-patient_to_subtype[0][0]]+=\
            (p.get_syn_mutations(region).sum(axis=0)>1)[patient_to_subtype[:,2]]
        for af, ysi, depth in izip(aft, p.ysi, p.n_templates_dilutions):
            if ysi<1:
                continue
            pat_af = af[:,patient_to_subtype[:,2]]
            patient_consensus = pat_af.argmax(axis=0)
            ind = ref_ungapped&rare&(patient_consensus==consensus)&(ancestral==consensus)&(final==consensus)
            w = depth/(1.0+depth/300.0)
            combined_af_by_pat[pcode][:,patient_to_subtype[ind,0]-patient_to_subtype[0][0]] \
                        += w*pat_af[:,ind]

    return combined_af_by_pat, syn_nonsyn_by_pat


def collect_data(patient_codes, regions, reference):
    '''
    loop over regions and produce a dictionary that contains the frequencies,
    syn/nonsyn designations and mutation rates
    '''
    cov_min=1000
    combined_af_by_pat={}
    syn_nonsyn_by_pat={}
    consensus_mutation_rate={}
    patients = []
    with open('data/mutation_rate.pickle') as mutfile:
        from cPickle import load
        mutation_rates = load(mutfile)
        total_muts = {nuc:sum([x for mut, x in mutation_rates['mu'].iteritems() if mut[0]==nuc]) for nuc in 'ACGT'}

    for pcode in patient_codes:
        try:
            p = Patient.load(pcode)
            patients.append(p)
        except:
            print("Can't load patient", pcode)
    for region in regions:
        combined_af_by_pat[region], syn_nonsyn_by_pat[region] = collect_weighted_afs(region, patients, reference)
        consensus_mutation_rate[region] = np.array([total_muts[nuc] if nuc!='-' else np.nan for nuc in
                            reference.annotation[region].extract("".join(reference.consensus))])

    return {'af_by_pat':combined_af_by_pat, 'mut_rate':consensus_mutation_rate, 'syn_by_pat':syn_nonsyn_by_pat}


def process_average_allele_frequencies(data, regions, nbootstraps = 0, bootstrap_type='bootstrap', nstates=4):
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
        minor_af[region] = (combined_af[region][:nstates,:].sum(axis=0) - combined_af[region].max(axis=0))/(1e-6+combined_af[region][:nstates,:].sum(axis=0))
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
    if nbootstraps:
        return combined_af, combined_entropy, minor_af,combined_entropy_bs, minor_af_bs
    else:
        return combined_af, combined_entropy, minor_af


def entropy_scatter(region, within_entropy, synnonsyn, reference, fname = None):
    '''
    scatter plot of cross-sectional entropy vs entropy of averaged intrapatient frequencies
    '''
    xsS = np.array([reference.entropy[ii] for ii in reference.annotation[region]])
    ind = xsS>=0.000
    print(region)
    print("Pearson:", pearsonr(within_entropy[region][ind], xsS[ind]))
    rho, pval = spearmanr(within_entropy[region][ind], xsS[ind])
    print("Spearman:", rho, pval)

    plt.figure(figsize = (7,6))
    for ni, syn_ind, label_str in ((0, ~synnonsyn[region], 'nonsynymous'), (2,synnonsyn[region], 'synonymous')):
        ind = (xsS>=0.000)&syn_ind
        plt.scatter(within_entropy[region][ind]+.00003, xsS[ind]+.005, c=cols[ni], label=label_str, s=30)
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


def fraction_diverse(region, minor_af, synnonsyn, fname=None):
    '''
    cumulative figures of the frequency distributions
    '''
    plt.figure()
    for ni, ind, label_str in ((0, ~synnonsyn[region], 'nonsynonymous'), (2,synnonsyn[region], 'synonymous')):
        plt.plot(sorted(minor_af[region][ind]+0.00001), np.linspace(0,1,ind.sum()),
                label=label_str+' n='+str(np.sum(ind)))
    plt.xscale('log')
    plt.yscale('linear')
    plt.legend(loc=2, fontsize=fs*0.8)
    plt.xlabel('minor frequency X', fontsize=fs)
    plt.ylabel(r'fraction less diverse than X', fontsize=fs)
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)


def selcoeff_distribution(regions, minor_af, synnonsyn, mut_rates, fname=None, ref=None):
    '''
    produce figure of distribution of selection coefficients separately for
    synonymous and nonsynonymous sites.
    '''
    if ref is not None:
        if not hasattr(ref, 'fitness_cost'):
            ref.fitness_cost = np.zeros_like(ref.entropy)
    fig, axs = plt.subplots(1,2,sharey=True)
    #plt.title(region+' selection coefficients')
    if type(regions)==str:
        regions = [regions]
    for ni,ax,label_str in ((0,axs[0], 'synonymous'), (1,axs[1], 'nonsynonymous')):
        slist = []
        for region in regions:
            ind = synnonsyn[region] if label_str=='synonymous' else ~synnonsyn[region]
            slist.extend(mut_rates[region][ind]/(minor_af[region][ind]+0.0001))
        s = np.array(slist)
        s[s>=0.1] = 0.1
        s[s<=0.001] = 0.001
        if ref is not None:
            bg = ref.annotation[region].location.start
            ed = ref.annotation[region].location.end
            ref.fitness_cost[bg:ed][ind] = s
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
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)


def selcoeff_confidence(region, data, fname=None):
    '''
    bootstrap the selection coefficients and make distributions of the bootstrapped
    values for subsets of sites with a defined median. this should give an impression
    of how variable the estimates are. three such distributions are combined in one
    figure
    '''
    (combined_af, combined_entropy, minor_af,combined_entropy_bs, minor_af_bs) = \
        process_average_allele_frequencies(data, [region], nbootstraps=100, bootstrap_type='bootstrap')
    minor_af_array=np.array(minor_af_bs[region])
    qtiles = np.vstack([scoreatpercentile(minor_af_array, x, axis=0) for x in [25, 50, 75]])
    scb = (data['mut_rate'][region]/(0.0001+qtiles)).T
    sel_coeff_array = (data['mut_rate'][region]/(0.0001+minor_af_array))
    sel_coeff_array[sel_coeff_array<0.001]=0.001
    sel_coeff_array[sel_coeff_array>0.1]=0.1
    which_quantile = np.zeros(minor_af_array.shape[1], dtype=int)
    thres = [20,40,60,90]
    for i,(ql, qu) in enumerate(zip(thres[:-1], thres[1:])):
        sl,su=scoreatpercentile(scb[:,1], ql), scoreatpercentile(scb[:,1], ql+2)
        which_quantile[(scb[:,1]>=sl)&(scb[:,1]<su)]=i+1

    plt.figure(figsize = (8,6))
    for i in range(1,len(thres)):
        ind = which_quantile==i
        npoints = ind.sum()*sel_coeff_array.shape[0]
        #plt.hist(scb[ind,1], weights = np.ones(ind.sum(),dtype=float)/ind.sum(),
        #        bins=np.logspace(-3,-1,21), alpha=0.3, color=cols[i])
        plt.plot(np.median(scb[ind,1])*np.ones(2), [0,0.5], c=cols[i], lw=4)
        plt.hist(sel_coeff_array[:,ind].flatten(), weights =np.ones(npoints,dtype=float)/npoints,
                bins=np.logspace(-3,-1,21), alpha=0.7, color=cols[i])
    plt.xscale('log')
    plt.xlabel('selection coefficient', fontsize=fs)
    plt.ylabel('uncertainty distribution', fontsize=fs)
    plt.tick_params(labelsize=fs*0.8)
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)


def selcoeff_vs_entropy(regions,  minor_af, synnonsyn, mut_rate, reference, fname=None, smoothing = 'harmonic'):
    fig = plt.figure()
    ax=plt.subplot(111)
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
            s.append(mut_rate[region][ind]/(minor_af[region][ind]+0.0001))
            entropy.append(xsS[ind])

        s = np.concatenate(s)
        entropy = np.concatenate(entropy)
        if label_str!='all':
            ax.scatter(entropy, s, c=cols[ni])

        A = np.array(sorted(zip(entropy+0.001, s), key=lambda x:x[0]))
        #ax.plot(np.exp(np.convolve(np.log(A[:,0]), 1.0*np.ones(npoints)/npoints, mode='valid')),
        #            np.exp(np.convolve(np.log(A[:,1]), 1.0*np.ones(npoints)/npoints, mode='valid')), c=cols[ni], label=label_str, lw=3)
        #ax.plot(np.convolve(A[:,0], 1.0*np.ones(npoints)/npoints, mode='valid'),
        #        np.convolve(A[:,1], 1.0*np.ones(npoints)/npoints, mode='valid'), c=cols[ni], label=label_str, lw=3)

        entropy_thresholds =  np.array(np.linspace(0,A.shape[0],8), int)
        entropy_boundaries = zip(entropy_thresholds[:-1], entropy_thresholds[1:])
        if smoothing=='harmonic':
            avg_sel_coeff[label_str] = np.array([(np.median(A[li:ui,0]), 1.0/np.mean(1.0/A[li:ui,1], axis=0))
                                                 for li,ui in entropy_boundaries])
            avg_sel_coeff[label_str+'_std'] = np.array([(np.median(A[li:ui,0]), 1.0/np.std(1.0/A[li:ui,1], axis=0)/np.sqrt(ui-li))
                                                        for li,ui in entropy_boundaries])
        elif smoothing=='median':
            avg_sel_coeff[label_str] = np.array([np.median(A[li:ui,:], axis=0) for li,ui in entropy_boundaries])
            avg_sel_coeff[label_str+'_std'] = np.array([(np.median(A[li:ui,0]), np.std(A[li:ui,1], axis=0)/np.sqrt(ui-li))
                                                        for li,ui in entropy_boundaries])
        elif smoothing=='geometric':
            avg_sel_coeff[label_str] = np.array([(np.median(A[li:ui,0]), np.exp(np.mean(np.log(A[li:ui,1]), axis=0)))
                                                 for li,ui in entropy_boundaries])
            avg_sel_coeff[label_str+'_std'] = np.array([(np.median(A[li:ui,0]), np.exp(np.std(np.log(A[li:ui,1], axis=0))/np.sqrt(ui-li)))
                                                        for li,ui in entropy_boundaries])

        ax.plot(avg_sel_coeff[label_str][:,0], avg_sel_coeff[label_str][:,1], lw=3)

    ax.legend()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('Fitness cost')
    ax.set_xlabel('Variability in group M [bits]')

    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)

    return avg_sel_coeff



# Script
if __name__=="__main__":

    parser = argparse.ArgumentParser(description='nucleotide allele frequencies, saturation levels, and fitness costs')
    parser.add_argument('--regenerate', action='store_true',
                        help="regenerate data")
    parser.add_argument('--subtype', choices=['B', 'any'], default='B',
                        help='subtype to compare against')
    args = parser.parse_args()

    reference = HIVreference(refname='NL4-3', subtype=args.subtype)

    fn = 'data/avg_nucleotide_allele_frequency.pickle.gz'

    regions = ['gag', 'pol', 'nef']
    if not os.path.isfile(fn) or args.regenerate:
        #patient_codes = ['p1', 'p2','p3','p5','p6', 'p8', 'p9','p10', 'p11'] # all subtypes, no p4/7
        #patient_codes = ['p1', 'p2','p3','p4', 'p5','p6','p7', 'p8', 'p9','p10', 'p11'] # patients
        patient_codes = ['p2','p3','p4','p5','p7', 'p8', 'p9','p10', 'p11'] # subtype B only
        data = collect_data(patient_codes, regions, reference)
        with gzip.open(fn, 'w') as ofile:
            cPickle.dump(data, ofile)
    else:
        with gzip.open(fn) as ifile:
            data = cPickle.load(ifile)
    if not all([region in data['mut_rate'] for region in regions]):
        print("data loading failed or data doesn't match specified regions:",
              regions, ' got:',data['mut_rate'].keys())

    combined_af, combined_entropy, minor_af = process_average_allele_frequencies(data, regions, nbootstraps=0)

    synnonsyn = {region: 2*np.array([x for x in data['syn_by_pat'][region].values()]).sum(axis=0)>len(data['syn_by_pat'][region])
                 for region in regions}

    for region in regions:
        entropy_scatter(region, combined_entropy, synnonsyn, reference, 'figures/'+region+'_entropy_scatter.png')
        fraction_diverse(region, minor_af, synnonsyn, 'figures/'+region+'_minor_allele_frequency.pdf')

        selcoeff_distribution(region, minor_af, synnonsyn, data['mut_rate'], 'figures/'+region+'_sel_coeff.png', ref=reference)
        selcoeff_confidence(region, data, 'figures/'+region+'_sel_coeff_confidence.png')

    selcoeff_distribution(regions, minor_af, synnonsyn,data['mut_rate'], 'figures/all_sel_coeff.png')

    avg_sel_coeff = selcoeff_vs_entropy(regions,  minor_af, synnonsyn,data['mut_rate'],
                                        reference,
                                        fname='figures/'+region+'_sel_coeff_scatter.png',
                                        smoothing='harmonic')

    with open('data/combined_af_avg_selection_coeff.pkl', 'w') as ofile:
        cPickle.dump(avg_sel_coeff, ofile)
