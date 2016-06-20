# vim: fdm=indent
'''
author:     Richard Neher
date:       21/03/16
content:    Combine allele frequencies from all patients (amino acids)
'''
# Modules
from __future__ import division, print_function

import os
import sys
import argparse
import cPickle
import gzip
import numpy as np
import pandas as pd
from itertools import izip
from collections import defaultdict
from scipy.stats import spearmanr, scoreatpercentile, pearsonr
from random import sample
from matplotlib import pyplot as plt
import seaborn as sns

from hivevo.patients import Patient
from hivevo.sequence import alphaal
from hivevo.HIVreference import HIVreferenceAminoacid, HIVreference
from hivevo.af_tools import divergence
from util import add_panel_label
from combined_af import process_average_allele_frequencies, draw_genome, af_average, get_final_state, load_mutation_rates



# Globals
fs=16
ERR_RATE = 2e-3
WEIGHT_CUTOFF = 500
SAMPLE_AGE_CUTOFF = 2 # years

ls = {'gag':'-', 'pol':'--', 'nef':'-.'}
cols = sns.color_palette()

drug_muts = {'PI':{'offset': 56 - 1,
                    'mutations':  [('L', 24, 'I'), ('V', 32, 'I'), ('M', 46, 'IL'), ('I', 47, 'VA'),
                        ('G', 48, 'VM'), ('I', 50, 'LV'), ('I', 54, 'VTAM'), ('L', 76, 'V'),
                        ('V', 82, 'ATSF'), ('I', 84, 'V'), ('N', 88, 'S'), ('L', 90, 'M')]},
             'NRTI':{'offset':56 + 99 - 1,
                    'mutations': [('M', 41, 'L'),('K', 65, 'R'),('D', 67, 'N'),('K', 70, 'ER'),('L', 74, 'VI'),
                                ('Y', 115, 'F'),  ('M', 184,'VI'), ('L', 210,'W'), ('T', 215,'YF'), ('K', 219,'QE')]
                   },
             'NNRTI':{'offset':56 + 99 - 1,
                    'mutations': [('L', 100, 'I'),('K', 101, 'PEH'), ('K', 103,'N'),
                                ('V', 106, 'AM'),('E', 138, 'K'),('V', 179, 'DEF'), ('Y', 181, 'CIV'),
                                ('Y', 188, 'LCH'),('G',190,'ASEQ'), ('F', 227,'LC'), ('M', 230,'L')]
                   }
            }
protective_positions = {
    'gag':{'gag':[12, 26, 28,79,147, 242, 248, 264, 268, 340, 357, 397, 403, 437]},
    'nef':{'nef':[71,81, 83, 85, 92, 94, 102,105,116,120,126,133,135]},
    'pol':{
    'PR':[35,93],
    'RT':[135, 245,277,369,395],
    'INT':[11,32,119,122,124],},
    'env':{'gp120':[], 'gp41':[]},
    'vif':{'vif':[]},
}
offsets = {
    'gag':-1,
    'PR':55,
    'nef':-1,
    'RT':55+99,
    'INT':55+99+440+120,
    'gp120':-1,
    'gp41':-1+481,
    'env':-1,
    'vif':-1
}


def aminoacid_mutation_rate(initial_codon, der, nuc_muts, doublehit=False):
    from Bio.Seq import CodonTable
    CT = CodonTable.standard_dna_table.forward_table
    targets = [x for x,a in CT.iteritems() if a==der]
    #print(CT[initial_codon], initial_codon, targets)
    mut_rate=0
    for codon in targets:
        nmuts = sum([a!=d for a,d in zip(initial_codon, codon)])
        mut_rate+=np.prod([nuc_muts[a+'->'+d] for a,d in
                    zip(initial_codon, codon) if a!=d])*((nmuts==1) or doublehit)
        #print(mut_rate, [nuc_muts[a+'->'+d] for a,d in zip(initial_codon, codon) if a!=d])

    return mut_rate


def calc_amino_acid_mutation_rates():
    from Bio.Seq import CodonTable

    nuc_mutation_rates = load_mutation_rates()['mu']

    CT = CodonTable.standard_dna_table.forward_table
    aa_mutation_rates = defaultdict(float)
    total_mutation_rates = defaultdict(float)
    for codon in CT:
        aa1 = CT[codon]
        for aa2 in alphaal:
            if aa1!=aa2:
                aa_mutation_rates[(codon,aa2)] += aminoacid_mutation_rate(codon, aa2, nuc_mutation_rates)

    for codon,aa1 in CT.iteritems():
        for pos in range(3):
            for nuc in 'ACTG':
                new_codon= codon[:pos]+nuc+codon[(pos+1):]
                if new_codon in CT:
                    if aa1!=CT[new_codon]:
                        total_mutation_rates[codon]+=nuc_mutation_rates[codon[pos]+'->'+nuc]
    return aa_mutation_rates, total_mutation_rates


def collect_weighted_aa_afs(region, patients, reference, cov_min=1000, max_div=0.05):
    '''
    produce weighted averages of allele frequencies for all late samples in each patients
    restrict to sites that don't sweep and have limited diversity as specified by max_div
    '''
    combined_af_by_pat = {}
    initial_codons_by_pat={}
    combined_phenos = {'disorder':np.zeros(len(reference.entropy)),
                        'accessibility':np.zeros(len(reference.entropy)),
                        'structural':np.zeros(len(reference.entropy))}

    good_pos_in_reference = reference.get_ungapped(threshold = 0.05)
    for pi, p in enumerate(patients):
        pcode = p.name
        combined_af_by_pat[pcode] = np.zeros(reference.af.shape)
        aft = p.get_allele_frequency_trajectories(region, cov_min=cov_min, type='aa', error_rate=ERR_RATE)

        # get patient to subtype map and initial aa and nuc sequence
        patient_to_subtype = p.map_to_external_reference_aminoacids(region, refname = reference.refname)
        init_nuc_sec = "".join(p.get_initial_sequence(region))
        consensus = reference.get_consensus_indices_in_patient_region(patient_to_subtype)
        ref_ungapped = good_pos_in_reference[patient_to_subtype[:,0]]

        # remember the codon at each reference position to be able to calculate mutation rates later
        initial_codons_by_pat[pcode] = {ci:init_nuc_sec[ii*3:(ii+1)*3] for ci, ii in patient_to_subtype}

        ancestral = p.get_initial_indices(region, type='aa')[patient_to_subtype[:,1]]
        rare = ((aft[:,:21,:]**2).sum(axis=1).min(axis=0)>max_div)[patient_to_subtype[:,1]]
        #final = aft[-1].argmax(axis=0)[patient_to_subtype[:,1]]
        final = get_final_state(aft[:,:,patient_to_subtype[:,1]])

        do=[]
        acc=[]
        struct=[]
        for pos in p.annotation[region]:
            if pos%3==1: # extract phenotypes for each
                try:
                    do.append(np.mean(p.pos_to_feature[pos]['disorder'].values()))
                except:
                    do.append(None)
                try:
                    struct.append(np.mean(p.pos_to_feature[pos]['structural'].values()))
                except:
                    struct.append(None)
                try:
                    acc.append(np.mean(p.pos_to_feature[pos]['accessibility'].values()))
                except:
                    acc.append(None)
        do = np.array(map(lambda x:0.0 if x is None else x, do))
        combined_phenos['disorder'][patient_to_subtype[:,0]]+= do[patient_to_subtype[:,1]]

        acc = np.array(map(lambda x:0.0 if x is None else x, acc))
        combined_phenos['accessibility'][patient_to_subtype[:,0]]+= acc[patient_to_subtype[:,1]]

        struct = np.array(map(lambda x:0.0 if x is None else x, struct))
        combined_phenos['structural'][patient_to_subtype[:,0]]+= struct[patient_to_subtype[:,1]]

        for af, ysi, depth in izip(aft, p.ysi, p.n_templates_dilutions):
            if ysi<SAMPLE_AGE_CUTOFF:
                continue
            pat_af = af[:,patient_to_subtype[:,1]]
            patient_consensus = pat_af.argmax(axis=0)
            ind = ref_ungapped&rare&(patient_consensus==consensus)&(ancestral==consensus)&(final==consensus)
            if pat_af.mask.any():
                ind = ind&(~pat_af.mask.any(axis=0))
            if ind.sum()==0:
                continue
            w = depth/(1.0+depth/WEIGHT_CUTOFF)
            combined_af_by_pat[pcode][:,patient_to_subtype[ind,0]] \
                        += w*pat_af[:-1,ind]
    for pheno in combined_phenos:
        combined_phenos[pheno]/=len(patients)
    return combined_af_by_pat, initial_codons_by_pat, combined_phenos


def collect_data(patient_codes, regions, subtype):
    cov_min=1000
    combined_af_by_pat={}
    initial_codons_by_pat={}
    combined_phenos={}
    aa_ref = 'NL4-3'
    patients = []

    for pcode in patient_codes:
        try:
            p = Patient.load(pcode)
            patients.append(p)
        except:
            print("Can't load patient", pcode)

    for region in regions:
        reference = HIVreferenceAminoacid(region, refname=aa_ref, subtype = subtype)
        combined_af_by_pat[region], initial_codons_by_pat[region], combined_phenos[region] =\
            collect_weighted_aa_afs(region, patients, reference, cov_min=cov_min)

    return {'af_by_pat':combined_af_by_pat, 'init_codon': initial_codons_by_pat, 'pheno':combined_phenos}


def get_optimal_epitopes(region, reference):
    epi_map = np.zeros_like(reference.entropy, dtype=int)
    lanl_optimal = pd.read_csv("../data/optimal_ctl_summary.csv")
    #lanl_optimal = pd.read_csv("../data/ctl_variant.csv")
    for ii, epi in lanl_optimal.iterrows():
        try:
            if ((epi.Protein.lower() == region) or (epi.Protein=='gp160' and region=='env')):
                if type(epi.loc['HLA']) is str:
                    if any([x in epi.loc['HLA'] for x in ['B57', 'B*57']]):
                        epi_map[epi.loc['HXB2 start']-1:epi.loc['HXB2 end']]+=1
        except:
            print(ii, epi)

    return epi_map


def get_associations(regions, aa_ref='NL4-3'):
    hla_assoc = pd.read_csv("../data/Carlson_el_al_2012_HLAassoc.csv")
    #hla_assoc = pd.read_csv("data/Brumme_et_al_HLAassoc.csv")
    qvalue_cutoff = 0.1
    associations = {}
    for region in regions:
        reference = HIVreferenceAminoacid(region, refname=aa_ref, subtype = 'B')
        L = len(reference.entropy)
        associations[region]={}
        if region=='env':
            subset = (hla_assoc.loc[:,"Protein"]=='gp120')&(hla_assoc.loc[:,"QValue"]<qvalue_cutoff)
            A = np.in1d(np.arange(L), np.unique(hla_assoc.loc[subset, "Position"])-1)
            subset = (hla_assoc.loc[:,"Protein"]=='gp41')&(hla_assoc.loc[:,"QValue"]<qvalue_cutoff)
            B = np.in1d(np.arange(L), np.unique(hla_assoc.loc[subset, "Position"]) + offsets['gp41'])
            hla_assoc_pos = A|B
        else:
            subset = (hla_assoc.loc[:,"Protein"]==region)&(hla_assoc.loc[:,"QValue"]<qvalue_cutoff) \
                      &np.array(["B*57" in x for x in hla_assoc.loc[:,"HLA"]], dtype=bool)
            if region=="pol":
                hla_assoc_pos = np.in1d(np.arange(L), np.unique(hla_assoc.loc[subset, "Position"])-1+56)
            else:
                hla_assoc_pos = np.in1d(np.arange(L), np.unique(hla_assoc.loc[subset, "Position"])-1)
        associations[region]['HLA'] = hla_assoc_pos
        ppos = []
        for feat, positions in protective_positions[region].iteritems():
            for pos in positions:
                ppos.append(pos+offsets[feat])
        associations[region]['protective'] = np.in1d(np.arange(L), np.unique(ppos))
    return associations


def fitness_costs_in_optimal_epis(regions, s):
    cols = sns.color_palette()
    plt.figure(figsize=(8,6))
    ax=plt.subplot(111)
    for ri, region in enumerate(regions):
        reference = HIVreferenceAminoacid(region, refname=aa_ref, subtype = args.subtype)
        epi = get_optimal_epitopes(region, reference)
        ind = ~np.isnan(s[region])
        print(region, 'fitness costs:', np.median(s[region][ind&(epi>0)]), np.median(s[region][ind&(epi==0)]))
        print(region, 'entropy:', np.median(reference.entropy[ind&(epi>0)]), np.median(reference.entropy[ind&(epi==0)]))

        stmp = np.copy(s[region])
        stmp[stmp<0.001]=0.001
        stmp[stmp>0.1]=0.1
        vals = stmp[ind&(epi>0)]
        plt.plot(sorted(vals), np.linspace(0,1,len(vals)), label=region+ ', A-list epitopes', c=cols[ri], lw=2)
#        plt.hist(vals, bins=np.logspace(-3,-1,21), weights = np.ones(len(vals), dtype=float)/len(vals),
#                 label='within A-list epitopes', alpha=0.5)
        vals = stmp[ind&(epi==0)]
#        plt.hist(vals, bins=np.logspace(-3,-1,21), weights = np.ones(len(vals), dtype=float)/len(vals),
#                 label='other', alpha=0.5)
        plt.plot(sorted(vals), np.linspace(0,1,len(vals)), label=region+ ', outside epitopes', ls='--', c=cols[ri], lw=2)

    plt.ylabel('fraction cost<X', fontsize=fs)
    plt.xlabel('fitness cost', fontsize=fs)
    plt.xscale('log')
    ax.set_xticks([0.001, 0.01, 0.1])
    ax.set_xticklabels([r'$<10^{-3}$', r'$10^{-2}$', r'$>10^{-1}$'])
    ax.tick_params(labelsize=0.8*fs)
    plt.legend(fontsize=0.8*fs, loc=2)
    plt.tight_layout()
    #add_panel_label(ax,'E', x_offset=-.2)
    for ext in ['png', 'svg', 'pdf']:
        plt.savefig('../figures/optimal_epitopes_st_'+args.subtype+'.'+ext)



def fitness_scatter(region, s, associations,
                    reference, annotate_protective=True, fname = None, running_avg=True):

    enrichment, rho, pval = entropy_scatter(region, s, associations, reference, fname=fname,
                            annotate_protective=annotate_protective,
                            running_avg=True, xlabel='fitness cost', xlim = 1e-4)
    return  enrichment, rho, pval

def entropy_scatter(region, within_entropy, associations, reference,
                fname = None, annotate_protective=False, running_avg=True,
                xlabel='pooled within patient entropy', xlim=1e-5):
    '''
    scatter plot of cross-sectional entropy vs entropy of averaged
    intrapatient frequencies amino acid frequencies
    '''
    xsS = reference.entropy
    ind = (xsS>=0.000)&(~np.isnan(within_entropy[region]))
    print(region)
    print("Pearson:", pearsonr(within_entropy[region][ind], xsS[ind]))
    rho, pval = spearmanr(within_entropy[region][ind], xsS[ind])
    print("Spearman:", rho, pval)

    plt.figure(figsize = (7,6))
    npoints=20
    assoc_ind = associations[region]['HLA']|associations[region]['protective']
    thres_xs = [0.0, 0.1, 10.0]
    thres_xs = zip(thres_xs[:-1],thres_xs[1:])
    nthres_xs=len(thres_xs)
    thres_in = [0.0, 0.0001, 10.0]
    thres_in = zip(thres_in[:-1],thres_in[1:])
    nthres_in=len(thres_in)
    enrichment = np.zeros((2,nthres_in, nthres_xs), dtype=int)
    for ni, assoc_ind, label_str in ((0, ~assoc_ind, 'other'), (2, assoc_ind, 'HLA/protective')):
        tmp_ind = assoc_ind&ind
        plt.scatter(within_entropy[region][tmp_ind]+.00003, xsS[tmp_ind]+.005, c=cols[ni], label=label_str, s=30)
        #plt.plot(sorted(within_entropy[region][tmp_ind]+.00003), np.linspace(0,1,tmp_ind.sum()), c=cols[ni], label=label_str)
        if running_avg: # add a running average to the scatter plot averaging over npoints on both axis
            A = np.array(sorted(zip(within_entropy[region][tmp_ind]+0.0000, xsS[tmp_ind]+0.005), key=lambda x:x[0]))
            plt.plot(np.exp(np.convolve(np.log(A[:,0]), np.ones(npoints, dtype=float)/npoints, mode='valid')),
                        np.exp(np.convolve(np.log(A[:,1]), np.ones(npoints, dtype=float)/npoints, mode='valid')),
                        c=cols[ni], lw=3)
            for ti1,(tl1, tu1) in enumerate(thres_in):
                for ti2,(tl2, tu2) in enumerate(thres_xs):
                    enrichment[ni>0, ti1, ti2] = np.sum((A[:,0]>=tl1)&(A[:,0]<tu1)&(A[:,1]>=tl2)&(A[:,1]<tu2))

    from scipy.stats import fisher_exact
    print(enrichment, fisher_exact(enrichment[:,:,1]))
    # add labels to points of positions of interest (positions with protective variation)
    if annotate_protective:
        A = np.array((within_entropy[region]+0.00003, xsS+0.005)).T
        for feat, positions in protective_positions[region].iteritems():
            for pos in positions:
                intra, cross = A[pos+offsets[feat],:]
                plt.annotate(feat+':' +str(pos), (intra, cross), (intra*1.05, cross*1.05), color='r')

    plt.ylabel('cross-sectional entropy', fontsize=fs)
    plt.xlabel(xlabel, fontsize=fs)
    plt.text(xlim*2, 3, r"Combined Spearman's $\rho="+str(round(rho,2))+"$", fontsize=fs)
    plt.legend(loc=4, fontsize=fs*0.8)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim([0.001, 4])
    plt.xlim([xlim, 1.0])
    plt.tick_params(labelsize=fs*0.8)
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)

    return enrichment, rho, pval


def phenotype_scatter(region, within_entropy, phenotype, phenotype_name, fname = None):
    '''
    scatter minor within patient variation against phenotypes associated to positions in proteins
    such as disorder scores, solvent accessible area or ddG calcuations. the values come from
    Li et al Retrovirology 2015 and Carlson, personal communication
    '''
    ind = (phenotype!=0)&(~np.isnan(within_entropy[region]))
    print(region, phenotype_name)
    print("Pearson:", pearsonr(within_entropy[region][ind], phenotype[ind]))
    rho, pval = spearmanr(within_entropy[region][ind], phenotype[ind])
    print("Spearman:", rho, pval)

    plt.figure(figsize = (7,6))
    plt.title("Spearman's rho: "+str(np.round(rho,2)))
    plt.scatter(within_entropy[region][ind]+.00003, phenotype[ind], s=30)

    plt.xlabel('pooled within patient entropy', fontsize=fs)
    plt.ylabel(phenotype_name, fontsize=fs)
    plt.xscale('log')
    plt.xlim([0.00001, .3])
    plt.tick_params(labelsize=fs*0.8)
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)

    return rho,pval


def correlation_vs_npat(pheno, region, data, reference):
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
    if pheno=='entropy':
        xsS = reference.entropy+1e-10
    else:
        xsS = np.array(data['pheno'][region][pheno])

    for n in range(1,1+N):
        for ii in range(int(min(2*binom(N,n),20))):
            subset = [data['af_by_pat'][region][x] for x in sample(pats, n)]
            tmp_af = af_average(subset)
            withS = -(np.log2(tmp_af+1e-10)*tmp_af).sum(axis=0)
            ind = (xsS>0.000)&(~np.isnan(withS))
            within_cross_correlation[n].append(spearmanr(xsS[ind], withS[ind])[0])

    return within_cross_correlation


def PhenoCorr_vs_Npat(pheno, data, figname=None, label_str=''):
    '''
    calculate cross-sectional and within patient entropy correlations
    for many subsets of patients and plot the average rank correlation
    against the number of patients used in the within patient average
    '''
    from util import add_panel_label
    plt.figure()
    for region in ['gag', 'pol', 'vif', 'nef']:
        reference = HIVreferenceAminoacid(region, refname=aa_ref, subtype = args.subtype)
        xsS_within_corr = correlation_vs_npat(pheno, region, data, reference)
        npats = sorted(xsS_within_corr.keys())
        avg_corr = [np.mean(xsS_within_corr[i]) for i in npats]
        std_corr = [np.std(xsS_within_corr[i]) for i in npats]
        plt.errorbar(np.array([1.0/i for i in npats]), y=avg_corr,yerr=std_corr, label=region)
    plt.legend(fontsize=fs)
    pheno_label = 'intra-patient/global entropy' if pheno=='entropy' \
                    else 'intra-patient entropy/'+pheno
    plt.ylabel(pheno_label+' correlation', fontsize=fs)
    plt.xlabel('1/number of patients', fontsize=fs)
    plt.xlim(0,1.1)
    plt.tick_params(labelsize=fs*0.8)
    add_panel_label(plt.gca(), label_str, x_offset=-0.10)
    plt.tight_layout()
    if figname is not None:
        for ext in ['png', 'svg', 'pdf']:
            plt.savefig(figname+'.'+ext)


def selection_coefficient_mutation(region, data, aa_mutation_rates, pos, target_aa, nbootstraps=0):
    '''
    determine the fitness cost associated with a particular amino acid mutations such as K103N
    this requires specification of the target amino acid and a specific calculation of
    the mutation rate into the amino acid, which requires the ancestral codon present in
    each individual patient
    '''
    def s(pats):
        # calcute nu/mu for each patient with patient specific mutation rates excluding double hit mutations
        nu_over_mu = [minor_af_by_pat[pat]/aa_mutation_rates[(codons[pat][pos],target_aa)] for pat in pats
                     if aa_mutation_rates[(codons[pat][pos],target_aa)]>0]
        # return the inverse, i.e. essentially the harmonic mean
        if len(nu_over_mu):
            savg = 1.0/max(0.01, np.mean(nu_over_mu))
        else:
            savg=np.nan
        return savg

    target_ii = alphaal.index(target_aa)
    codons = data['init_codon'][region]
    minor_af_by_pat = {pat: x[target_ii,pos].sum(axis=0)/x[:20,pos].sum(axis=0)
                        for pat, x in data['af_by_pat'][region].iteritems() if pos in codons[pat]}
    all_patients = minor_af_by_pat.keys()

    if nbootstraps:
        s_bs = []
        for bi in xrange(nbootstraps):
            tmp_s = s([all_patients[pi] for pi in np.random.randint(len(all_patients), size=len(all_patients))])
            if not np.isnan(tmp_s):
                s_bs.append(tmp_s)
        if len(s_bs):
            s_out = [np.percentile(s_bs, perc) for perc in [5, 25, 50, 75,95]]
        else:
            s_out = [np.nan for perc in [5, 25, 50, 75,95]]
    else:
        s_out = s(all_patients)
    return s_out


def selection_coefficients_per_site(region,data, total_nonsyn_mutation_rates, nbootstraps=None):
    codons = data['init_codon'][region]
    minor_af_by_pat = {pat: (x[:20,:].sum(axis=0) - x[:20,:].max(axis=0))/x[:20,:].sum(axis=0)
                        for pat, x in data['af_by_pat'][region].iteritems()}

    if nbootstraps is None:
        pat_sets = [minor_af_by_pat.keys()]
    else:
        pats = minor_af_by_pat.keys()
        pat_sets = [[pats[ii] for ii in np.random.randint(len(pats), size=len(pats))] for jj in range(nbootstraps)]
    s_bs = []
    for pat_set in pat_sets:
        nu_over_mu = []
        for pat in pat_set:
            tmp=[]
            for pos, nu in enumerate(minor_af_by_pat[pat]):
                if pos in codons[pat]:
                    tmp.append(nu/total_nonsyn_mutation_rates[codons[pat][pos]])
                else:
                    tmp.append(np.nan)

            nu_over_mu.append(tmp)

        tmp_nu_over_mu = np.ma.array(nu_over_mu)
        tmp_nu_over_mu.mask = np.isnan(nu_over_mu)
        #import ipdb; ipdb.set_trace()
        tmp_s = 1.0/(tmp_nu_over_mu.mean(axis=0)+0.1)
        tmp_s[tmp_s.mask] = np.nan
        s_bs.append(tmp_s)
    if nbootstraps is None:
        return s_bs[-1]
    else:
        return np.array(s_bs)


def selection_coefficients_distribution(region, data, total_nonsyn_mutation_rates):
    selcoeff = selection_coefficients_per_site(region, data, total_nonsyn_mutation_rates)
    selcoeff[selcoeff<0.001]=0.001
    selcoeff[selcoeff>0.1]=0.1
    ind = ~np.isnan(selcoeff)
    n=ind.sum()
    if n>0:
        plt.figure(figsize=(8,6))
        plt.hist(selcoeff[ind], weights=np.ones(n, dtype=float)/n, bins=np.logspace(-3,-1,11))
        plt.xscale('log')
    else:
        print("NO SELECTION COEFFICENTS FOR", region)


def selection_coefficients_compare(regions, data, total_nonsyn_mutation_rates):
    '''
    compare the distribution of selection coefficients among different regions in the genome
    '''
    plt.figure()
    selcoeff = {}
    for ri,region in enumerate(regions):
        sc =  selection_coefficients_per_site(region, data, total_nonsyn_mutation_rates)
        sc[sc<0.001]=0.001
        sc[sc>0.1]=0.1
        selcoeff[region] = np.ma.array(sc, mask=np.isnan(sc))
        n=(selcoeff[region].mask==False).sum()
        if n>0:
            y,x = np.histogram(selcoeff[region].compressed(), weights=np.ones(n, dtype=float)/n, bins=np.logspace(-3,-1,11))
            plt.plot(np.sqrt(x[1:]*x[:-1]), y, label=region, c=cols[ri], lw=3)
        else:
            print("NO SELECTION COEFFICENTS FOR", region)

    plt.legend(loc=2)
    plt.xscale('log')
    from scipy.stats import ks_2samp
    for r1 in regions:
        for r2 in regions:
            print(r1,r2, ks_2samp(selcoeff[r1], selcoeff[r2]))

    return selcoeff


def selection_coefficients_compare_pheno(pheno, threshold, regions, data, total_nonsyn_mutation_rates, plot=False, cumulative=True):
    '''
    compare the distribution of selection coefficients between sites with a phenotype above of below
    the threshold. optionally plots the distribution
    '''
    if plot: plt.figure()
    selcoeff = {}
    for ri,region in enumerate(regions):
        sc =  selection_coefficients_per_site(region, data, total_nonsyn_mutation_rates)
        sc[sc<0.001]=0.001
        sc[sc>0.1]=0.1
        valid = data['pheno'][region][pheno]>0
        above = valid&(data['pheno'][region][pheno]>threshold)
        below = valid&(data['pheno'][region][pheno]<=threshold)
        print(above.sum(), below.sum())
        selcoeff[region] = (np.ma.array(sc[above], mask=np.isnan(sc[above])),
                            np.ma.array(sc[below], mask=np.isnan(sc[below])))
        if plot:
            n=len(selcoeff[region][0].compressed())
            if cumulative:
                ab = selcoeff[region][0].compressed()
                bl = selcoeff[region][1].compressed()
                plt.plot(sorted(ab), np.linspace(0,1,len(ab)), label=region+' below', ls='--', c=cols[ri], lw=3)
                plt.plot(sorted(bl), np.linspace(0,1,len(bl)), label=region+' above', ls='-', c=cols[ri], lw=3)
            else:
                y,x = np.histogram(selcoeff[region][0].compressed(), weights=np.ones(n, dtype=float)/n, bins=np.logspace(-3,-1,11))
                plt.plot(np.sqrt(x[1:]*x[:-1]), y, label=region+' above', ls='--', c=cols[ri], lw=3)
                n=len(selcoeff[region][1].compressed())
                y,x = np.histogram(selcoeff[region][1].compressed(), weights=np.ones(n, dtype=float)/n, bins=np.logspace(-3,-1,11))
                plt.plot(np.sqrt(x[1:]*x[:-1]), y, label=region+' below', c=cols[ri], lw=3)


    if plot:
        plt.legend(loc=2)
        plt.xscale('log')
    from scipy.stats import ks_2samp
    KS = {}
    for r1 in regions:
        try:
            KS[r1] = ks_2samp(selcoeff[r1][0], selcoeff[r1][1])
            print(r1, KS[r1])
        except:
            print(r1, pheno, 'failed')
    return KS


def selection_coefficients_compare_association(association, associations, regions, data, total_nonsyn_mutation_rates):
    '''
    essentially the same as selection_coefficients_compare_pheno but for binary associations rather than
    continuous phenotypes.
    '''
    plt.figure()
    selcoeff = {}
    for ri,region in enumerate(regions):
        sc =  selection_coefficients_per_site(region, data, total_nonsyn_mutation_rates)
        sc[sc<0.001]=0.001
        sc[sc>0.1]=0.1
        above = associations[region][association]
        below = ~associations[region][association]
        print(above.sum(), below.sum())
        selcoeff[region] = (np.ma.array(sc[above], mask=np.isnan(sc[above])),
                            np.ma.array(sc[below], mask=np.isnan(sc[below])))
        n=len(selcoeff[region][0].compressed())
        y,x = np.histogram(selcoeff[region][0].compressed(), weights=np.ones(n, dtype=float)/n, bins=np.logspace(-3,-1,11))
        plt.plot(np.sqrt(x[1:]*x[:-1]), y, label=region+' associated', ls='--', c=cols[ri], lw=3)
        n=len(selcoeff[region][1].compressed())
        y,x = np.histogram(selcoeff[region][1].compressed(), weights=np.ones(n, dtype=float)/n, bins=np.logspace(-3,-1,11))
        plt.plot(np.sqrt(x[1:]*x[:-1]), y, label=region+' non associated', c=cols[ri], lw=3)

    plt.legend(loc=2)
    plt.xscale('log')
    from scipy.stats import ks_2samp
    KS = {}
    for r1 in regions:
        KS[r1] = ks_2samp(selcoeff[r1][0], selcoeff[r1][1])
        print(r1, KS[r1])
    return KS


def compare_experiments(data, aa_mutation_rates):
    fc = pd.read_csv('../data/fitness_costs.csv')
    coefficients = {}
    for ii, mut in fc.iterrows():
        region = mut['region']
        offset = offsets[mut['feature']]
        aa, pos = mut['mutation'][-1], int(mut['mutation'][1:-1])+offset
        coefficients[(mut['feature'], mut['mutation'])] = (mut['normalized'],
            selection_coefficient_mutation(region, data, aa_mutation_rates, pos, aa, nbootstraps=100))

    return coefficients


def compare_hinkley(data, reference, total_nonsyn_mutation_rates, fname=None):
    from parse_hinkley import parse_hinkley
    cutoff=0.1
    hinkley_cost = {}
    hfit = parse_hinkley()
    selcoeff = selection_coefficients_per_site('pol', data, total_nonsyn_mutation_rates)
    ref_aa = np.array(reference.seq.seq.translate())
    past_cutoff = selcoeff>cutoff
    selcoeff[past_cutoff]=cutoff
    selcoeff[selcoeff<0.0003]=0.0003
    non_consensus = []
    for prot, pos in hfit:
        ref_pos = pos+offsets[prot]+1
        #consensus = reference.consensus[ref_pos]
        consensus = ref_aa[ref_pos]
        s = selcoeff[ref_pos]
        if consensus in hfit[(prot,pos)]:
            tmp_non_cons = [val for aa, val in hfit[(prot,pos)].iteritems() if aa!=consensus]
            non_consensus.extend(tmp_non_cons)
            if len(hfit[(prot,pos)])>1 and (not past_cutoff[ref_pos]):
                gaps = np.diff(sorted(hfit[(prot,pos)].values(),reverse=True))
                hinkley_cost[(prot, pos)] = (s, hfit[(prot,pos)][consensus],\
                    np.max(tmp_non_cons), -gaps[0])
        else:
            print('Not found',consensus,' in:', prot,pos,hfit[(prot,pos)].keys())

    s_array = np.array(hinkley_cost.values())
    plt.figure()
    plt.scatter(s_array[:,0], s_array[:,1]-s_array[:,2])
    plt.xscale('log')
    plt.xlim([0.001, 0.2])
    plt.xlabel('fitness cost')
    plt.ylabel('Hinkley fitess effect')
    print(spearmanr(s_array[:,0], s_array[:,1]-s_array[:,2]))
    if fname is not None:
        plt.savefig(fname)

    return hinkley_cost


def plot_drug_resistance_mutations(data, aa_mutation_rates, fname=None):
    '''Plot the frequency of drug resistance mutations'''
    import matplotlib.patches as patches

    region='pol'
    pcodes = data['init_codon'][region].keys()

    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 6]})
    ax = axs[1]

    drug_afs_items = []
    mut_types = []
    drug_classes = ['PI', 'NRTI', 'NNRTI']
    for prot in drug_classes:
        drug_afs = {}
        drug_mut_rates = {}
        offset = drug_muts[prot]['offset']
        for cons_aa, pos, target_aa in drug_muts[prot]['mutations']:
            codons = {pat:data['init_codon'][region][pat][pos+offset] for pat in pcodes}
            mut_rates = {pat:np.sum([aa_mutation_rates[(codons[pat], aa)] for aa in target_aa])
                        for pat in pcodes}
            freqs = {pat:np.sum([data['af_by_pat'][region][pat][alphaal.index(aa), pos+offset]\
                                /data['af_by_pat'][region][pat][:20,pos+offset].sum()
                        for aa in target_aa]) for pat in pcodes}


            drug_afs[(cons_aa,pos,target_aa)] = freqs
            drug_mut_rates[(cons_aa,pos,target_aa)] = mut_rates

        drug_afs_items.extend(filter(lambda x:np.sum(filter(lambda y:~np.isnan(y), x[1].values()))>0,
                                     sorted(drug_afs.items(), key=lambda x:x[0][1])))
        mut_types.append(len(drug_afs_items))
        #make list of all mutations whose non-nan frequencies sum to 0
        mono_muts = [''.join(map(str,x[0])) for x in
                    filter(lambda x:np.sum(filter(lambda y:~np.isnan(y), x[1].values()))==0,
                           sorted(drug_afs.items(), key=lambda x:x[0][1]))]
        print('Monomorphic:', prot, mono_muts)

    plt.ylim([1.1e-5, 1e-1])
    for mi in mut_types[:-1]:
        plt.plot([mi-0.5,mi-0.5], plt.ylim(), c=(.3,.3,.3), lw=3, alpha=0.5)
    ax.axhline(4e-5, c=(.3, .3, .3), lw=3, alpha=0.5)

    for ni, prot in enumerate(drug_classes):
        plt.text(0.5*(mut_types[ni] + (mut_types[ni-1] if ni else 0))-0.5, 0.12,
                 prot, fontsize=16, ha='center')

    for mi in range(max(mut_types)):
        c = 0.5 + 0.2*(mi%2)
        ax.add_patch( patches.Rectangle(
                (mi-0.5, plt.ylim()[0]),  1.0, plt.ylim()[1], #(x,y), width, height
                color=(c,c,c), alpha=0.2
            )
        )


    #plt.xticks(np.arange(len(all_muts)), ["".join(map(str, x)) for x in all_muts], rotation=60)
    afdr = pd.DataFrame(np.array([x[1].values() for x in drug_afs_items]).T,
                        columns=["".join(map(str,x[0])) for x in drug_afs_items])
    afdr[afdr < 0.8e-4] = 0
    sns.stripplot(data=afdr, jitter=0.4, alpha=0.8, size=12, lw=1, edgecolor='white')

    # Add the number of missing points at the bottom of the plot, and the cost
    # at the top
    dd = afdr.iloc[[0, 1, 2, 3]].copy()
    dd.index = ['x', 'freq', 'size', 'cost']
    dd.loc['x'] = np.arange(dd.shape[1])
    dd.loc['freq'] = 2e-5
    dd.loc['n'] = afdr.shape[0] - (afdr > 1e-4).sum(axis=0)
    dd.loc['size'] = dd.loc['n']**(1.4) + 13
    dd.loc['cost'] = 1.0 / afdr.fillna(0).mean(axis=0)
    # NOTE: the first 6 mutations are in PR, the rest in RT
    import re
    from Bio.Seq import translate
    reference = HIVreference(refname='HXB2', load_alignment=False)
    seq_PR = reference.annotation['PR'].extract(reference.seq)
    seq_RT = reference.annotation['RT'].extract(reference.seq)
    murate = load_mutation_rates()['mu']
    for i, mut in enumerate(dd.T.index):
        mr = 0
        if i < 6:
            seq_tmp = seq_PR
        else:
            seq_tmp = seq_RT
        aa_from, pos, aas_to = re.sub('([A-Z])(\d+)([A-Z]+)', r'\1_\2_\3', mut).split('_')
        cod = str(seq_tmp.seq[(int(pos) - 1) * 3: int(pos) * 3])
        for pos_cod in xrange(3):
            for nuc in ['A', 'C', 'G', 'T']:
                codmut = list(cod)
                codmut[pos_cod] = nuc
                codmut = ''.join(codmut)
                if (codmut != cod) and (translate(cod) == aa_from) and (translate(codmut) in aas_to):
                    mr += murate[cod[pos_cod]+'->'+nuc]

        dd.loc['cost', mut] *= mr

    for im, (mutname, s) in enumerate(dd.T.iterrows()):
        ax.scatter(s['x'], s['freq'],
                   s=s['size']**2,
                   alpha=0.8,
                   edgecolor='white',
                   facecolor=sns.color_palette('husl', afdr.shape[1])[im],
                   lw=2,
                  )
        ax.text(s['x'], s['freq'], str(int(s['n'])), fontsize=fs, ha='center', va='center')

    plt.yscale('log')
    plt.xticks(rotation=40)
    plt.ylabel('minor variant frequency', fontsize=fs)
    plt.tick_params(labelsize=fs*0.8)

    # Fitness cost at the top
    ax1 = axs[0]
    ax1.set_xlim(*ax.get_xlim())
    ax1.set_xticks(ax.get_xticks() + 0.5)
    ax1.set_xticklabels([])
    ax1.set_ylim(1e-3, 1)
    ax1.set_yticks([1e-3, 1e-2, 1e-1, 1])
    ax1.set_yscale('log')
    ax1.set_ylabel('cost', fontsize=fs)
    for im, (mut, y) in enumerate(dd.loc['cost'].iteritems()):
        ax1.bar(im - 0.5, y, 1, color=sns.color_palette('husl', afdr.shape[1])[im])

    plt.tight_layout()

    if fname is not None:
        for ext in ['svg', 'pdf', 'png']:
            plt.savefig(fname+'.'+ext)
    else:
        plt.ion()
        plt.show()


def export_selection_coefficients(data, total_nonsyn_mutation_rates, subtype):
    '''
    write selection coefficients as tab separated files
    files contain position and 25%, 50% and 75% of 100 bootstrap replicates
    '''
    from scipy.stats import scoreatpercentile
    def sel_out(s):
        if s<0.001:
            return '<0.001'
        elif s>0.1:
            return '>0.1'
        else:
            return s

    for region in data['af_by_pat']:
        reference = HIVreferenceAminoacid(region, refname=aa_ref, subtype = args.subtype)
        ref_seq = reference.seq.seq.translate()
        sel_array = selection_coefficients_per_site(region, data,
                            total_nonsyn_mutation_rates, nbootstraps=100)
        selcoeff={}
        for q in [25, 50, 75]:
            selcoeff[q] = scoreatpercentile(sel_array, q, axis=0)

        with open('../data/'+region+'_selection_coefficients_st_'+subtype+'.tsv','w') as selfile:
            selfile.write('### selection coefficients in '+region+'\n')
            selfile.write('\t'.join(['# position','consensus', reference.refname,
                                    'lower quartile','median','upper quartile'])+'\n')

            for pos in xrange(selcoeff[25].shape[0]):
                selfile.write('\t'.join(map(str,[pos+1, reference.consensus[pos], ref_seq[pos]]+
                        [sel_out(selcoeff[q][pos]) for q in [25, 50, 75]]))+'\n')


def plot_drug_resistance_mutation_trajectories(pcode):
    '''
    auxillary function to check for potential drug resistance evolution in RNA sequences
    only p10 has drug resistance mutations in the last two samples
    '''
    plt.figure()
    p = Patient.load(pcode)
    RT = p.get_allele_frequency_trajectories('RT', type='aa')
    for mt in ['NNRTI', 'NRTI']:
        for aa1, pos, aa2 in drug_muts[mt]['mutations']:
            traj =1-RT[:,alphaal.index(aa1), pos-1]
            if max(traj)>0.1:
                plt.plot(p.dsi, traj,'-o', label=mt+ ' '+str(pos))

    PR = p.get_allele_frequency_trajectories('PR', type='aa')
    for mt in ['PI']:
        for aa1, pos, aa2 in drug_muts[mt]['mutations']:
            traj =1-PR[:,alphaal.index(aa1), pos-1]
            if max(traj)>0.1:
                plt.plot(p.dsi, traj,'-o', label=mt+ ' '+str(pos))

    plt.legend(loc=2, ncol=2)



# Script
if __name__=="__main__":
    plt.ion()

    parser = argparse.ArgumentParser(description='amino acid allele frequencies, saturation levels, and fitness costs')
    parser.add_argument('--regenerate', action='store_true',
                        help="regenerate data")
    parser.add_argument('--subtype', choices=['B', 'any'], default='B',
                        help='subtype to compare against')
    args = parser.parse_args()

    fn = '../data/avg_aa_allele_frequency_st_'+args.subtype+'.pickle.gz'

    regions = ['gag', 'pol', 'nef', 'env', 'vif']
    if not os.path.isfile(fn) or args.regenerate:
        if args.subtype=='B':
            #patient_codes = ['p2','p3','p5','p8','p9','p10','p11'] # subtype B only
            patient_codes = ['p2','p3', 'p5','p7', 'p8','p9','p10', 'p11'] # patients
        else:
            patient_codes = ['p1','p2','p3', 'p5','p6','p7', 'p8','p9','p10', 'p11'] # patients
            #patient_codes = ['p1','p2','p3','p5','p6','p8','p9','p10', 'p11'] # patients
        data = collect_data(patient_codes, regions, args.subtype)
        with gzip.open(fn, 'w') as ofile:
            cPickle.dump(data, ofile)
    else:
        with gzip.open(fn) as ifile:
            data = cPickle.load(ifile)

    av = process_average_allele_frequencies(data, regions, nbootstraps=0,nstates=20)
    combined_af = av['combined_af']
    combined_entropy = av['combined_entropy']
    minor_af = av['minor_af']

    associations = get_associations(regions)
    aa_mutation_rates, total_nonsyn_mutation_rates = calc_amino_acid_mutation_rates()
    selcoeff = {region: selection_coefficients_per_site(region, data, total_nonsyn_mutation_rates, nbootstraps=None) for region in regions}

    plot_drug_resistance_mutations(data, aa_mutation_rates, '../figures/figure_5_subtype_'+args.subtype)

    aa_ref = 'NL4-3'
    global_ref = HIVreference(refname=aa_ref, subtype=args.subtype)

    fitness_costs_in_optimal_epis(['gag', 'nef'], selcoeff)

    phenotype_correlations = {}
    erich = np.zeros((2,2,2))
    for region in regions:
        reference = HIVreferenceAminoacid(region, refname=aa_ref, subtype = args.subtype)
        tmp, rho, pval =  fitness_scatter(region, selcoeff, associations, reference,
                                            fname='../figures/'+region+'_aa_fitness_scatter_st_'
                                                    +args.subtype+'.pdf')
        selection_coefficients_distribution(region, data, total_nonsyn_mutation_rates)

        if region == 'pol':
            compare_hinkley(data,reference, total_nonsyn_mutation_rates,
                            fname='../figures/hinkley_comparison_'+args.subtype+'.pdf')
#        tmp, rho, pval = entropy_scatter(region, combined_entropy, associations, reference,
#                                         '../'+region+'_aa_entropy_scatter_st_'
#                                        +args.subtype+'.pdf', annotate_protective=True)


        phenotype_correlations[(region, 'entropy')] = (rho, pval)
        erich+=tmp

        for phenotype, vals in data['pheno'][region].iteritems():
            try:
                print(phenotype)
                rho, pval = phenotype_scatter(region, selcoeff, vals, phenotype)
                phenotype_correlations[(region, phenotype)] = (rho, pval)
            except:
                print("Phenotype scatter failed for:",region, phenotype)
                phenotype_correlations[(region, phenotype)] = ('NaN', 'NaN')

    with open("../data/phenotype_correlation_st_"+args.subtype+".tsv", 'w') as pheno_file:
        pt = ['entropy', 'disorder', 'accessibility']
        pheno_file.write('\t'.join(['gene']+pt)+'\n')
        for region in regions:
            pheno_file.write('\t'.join([region]+
                [str(np.round(phenotype_correlations[(region, pheno)][0],3)) for pheno in pt])+'\n')

    export_selection_coefficients(data, total_nonsyn_mutation_rates, args.subtype)

    sc = selection_coefficients_compare(regions, data, total_nonsyn_mutation_rates)


    pval = []
    pheno='structural'
    for thres in np.linspace(0,3,21):
        KS = selection_coefficients_compare_pheno(pheno, thres, regions,
                                data, total_nonsyn_mutation_rates, plot=False)
        pval.append((thres, np.log(KS['pol'].pvalue)))

    pval = np.array(pval)
    plt.plot(pval[:,0], pval[:,1])

    KS_disorder = selection_coefficients_compare_pheno('disorder', 0.5, regions,
                                data, total_nonsyn_mutation_rates, plot =True)

    KS_accessibility = selection_coefficients_compare_pheno('accessibility', 75, regions,
                                data, total_nonsyn_mutation_rates, plot=True)

    KS_structural = selection_coefficients_compare_pheno('structural', 2.0, regions,
                                data, total_nonsyn_mutation_rates, plot =True)

    KS_HLA = selection_coefficients_compare_association('HLA', associations, regions,
                                data, total_nonsyn_mutation_rates)



#    fig, axs = plt.subplots(2,1,sharex=True, gridspec_kw={'height_ratios':[8, 1]})
#
#    ws=50
#    for ri, region in enumerate(regions):
#        #axs[0].plot([x for x in global_ref.annotation[region] if x%3==0], 1.0/np.convolve(np.ones(ws, dtype=float)/ws, 1.0/sc[region], mode='same'), c=cols[ri])
#        axs[0].plot([x for x in global_ref.annotation[region] if x%3==0], np.convolve(np.ones(ws, dtype=float)/ws, sc[region], mode='same'), c=cols[ri], ls='-')
#
#    draw_genome(axs[1], {k:val for k,val in global_ref.annotation.iteritems() if k in ['p17', 'p6', 'p7', 'p24', 'PR', 'RT', 'IN', 'p15', 'nef','gp41', 'vif']})

    PhenoCorr_vs_Npat('entropy', data,
                      figname='../figures/aa_entropy_corr_vs_patients_st_'+args.subtype,
                      label_str='C' if args.subtype=='any' else 'D')

