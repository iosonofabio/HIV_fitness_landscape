# vim: fdm=indent
'''
author:     Richard Neher
date:       22/02/16
content:    Combine allele frequencies from all patients for strongly conserved sites
            and analyze properties of non-coding features of the HIV-1 genome
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

from combined_af import process_average_allele_frequencies, draw_genome
from combined_af import af_average, load_mutation_rates, collect_data, running_average



# Globals
ERR_RATE = 2e-3
WEIGHT_CUTOFF = 500
SAMPLE_AGE_CUTOFF = 2 # years
af_cutoff = 1e-5
sns.set_style('darkgrid')
cols = sns.color_palette(n_colors=7)
fs = 16
#regions = ['RRE', 'psi', "LTR3'", "LTR5'"]
regions = ['genomewide', 'gag', 'nef']
plt.ion()

features = {
    'polyA':[(514,557)], # from LANL HXB2 annotation
    'U5':[(558,570)],    # from LANL, ref 10.1126/science.1210460
    'U5 stem':[(588,596),(624,631)],  # from LANL,
    'PBS':[(636,654)],  # from LANL,
    'interferon-stimulated response':[(655,674)],  # from LANL,
    'PSI SL1-4':[(691, 735), (736, 755), (766,779), (790, 811)],  # from LANL,
    'frameshift':[(2086,2093),(2101,2126)],  # from LANL,
    'Wang et al':[(2468,2578)],  # from Wang et al,
    'nuc_hyper':[(4400,4900)],  # from LANL,
    'siteC':[(4680,4700)],  # from LANL,
    'oct-1':[(4776,4790)],  # from LANL,
    'siteD':[(4816,4851)],  # from LANL,
    'A1':[(4887,4905)],  # Saliou et al,
    'D2':[(4959,4968)],  # Saliou et al,
    'cPPT':[(4785,4810)],  # from LANL,
    # RRE extracted from Siegfried et al NL4-3 -> HXB2 coordinates + 454 since their NL4-3 starts at TAR
    'RRE':[ (7780, 7792), (7796, 7805), (7808, 7813),(7817, 7825),(7829, 7838),(7846, 7853),
            (7856, 7863), (7865, 7872), (7875, 7880), (7886, 7892), (7901, 7913),
            (7916, 7928), (7931, 7940),(7948, 7957),(7962, 7972),(7975, 8003)],
    'PPT':[(9069,9094)],  # from LANL,
    'TCF-1alpha':[(9400,9415)],  # from LANL,
    'NK-kappa B1':[(9448,9459)],  # from LANL,
    'SP1':[(9462,9472),(9473,9483),(9483,9494)],  # from LANL,
    'TATA':[(9512,9517)],  # from LANL,
    'TAR':[(9538,9601)],  # from LANL,
}



# Functions
def plot_fitness_costs_along_genome(start, stop, feature_names,
                            data, minor_af,reference,synnonsyn=None,
                            ws=30, ws_syn=30, pheno=None, ax=None,
                            ybar=0.13, ytext=0.145):
    '''
    Plot the fitness costs along the genome in an interval specified by
    start and stop. Adds genome annotation of elements to feature_names
    optionally includes a running average
    '''
    from util import add_panel_label

    if ax is None:
        fig = plt.figure(figsize = (6,6))
        ax=plt.subplot(111)

    region = 'genomewide' # use genome wide in HXB2 coordinates

    ind = ~np.isnan(minor_af[region])
    sc = (data['mut_rate'][region]/(af_cutoff+minor_af[region]))
    sc[sc>0.1] = 0.1
    sc[sc<0.001] = 0.001
    # add individual data points
    ax.plot(np.arange(minor_af[region].shape[0])[ind],sc[ind],'o',
            ms=3,
            c=cols[0],
            alpha=0.5)

    # add a running average of selection coefficients
    ax.plot(running_average(np.arange(minor_af[region].shape[0])[ind], ws),
            np.exp(running_average(np.log(sc[ind]), ws)),
            c=cols[0], ls='-', label='all sites')

    # repeat running average, but restricted to sites that are synonymous
    if synnonsyn is not None:
        ind = (~np.isnan(minor_af[region]))&synnonsyn
        ax.plot(running_average(np.arange(minor_af[region].shape[0])[ind], ws),
                np.exp(running_average(np.log(sc[ind]), ws_syn)),
                c=cols[2], ls='-', label='non-coding/synonymous')

    # add features
    colors = sns.color_palette('husl', 7)
    for fi,feat in enumerate(feature_names):
        elements = features[feat]
        # add features as horizontal bars
        all_pos = []
        color = colors[(2 * fi) % len(colors)]
        for element in elements:
            all_pos.extend(element)
            ax.plot(np.array(element)-1, [ybar, ybar], lw=3,
                    color=color)

        ytext_tmp = ytext
        if (fi % 2) and (feat not in ['D2']):
            ytext_tmp *= 1.26
        ax.text((elements[0][0]+elements[-1][1])*0.5,
                ytext_tmp,
                feat,
                horizontalalignment='center',
                fontsize=fs*0.8)

    ax.set_yscale('log')
    ax.tick_params(labelsize=fs*0.8)
    ax.set_xlim(start, stop)
    for tl in ax.get_xmajorticklabels():
        tl.set_rotation(45)
    return ax


def plot_non_coding_figure(data, minor_af, synnonsyn, reference, fname=None):
    '''Plot fitness cost at noncoding features'''
    from util import add_panel_label

    ymax = 0.25
    ymin = 0.0005

    y_second_gene = 1.18

    fig, axs = plt.subplots(1, 4, sharey=True, figsize =(10,5),
                            gridspec_kw={'width_ratios':[4, 1, 2.5, 1]})

    # plot the 5' region
    start, stop = 500, 900
    feature_names = ['polyA', 'U5', 'U5 stem', 'PBS', 'PSI SL1-4']
    ax = plot_fitness_costs_along_genome(start, stop, feature_names, data,
                                     minor_af, reference, pheno=None,
                                     synnonsyn=synnonsyn['genomewide'],
                                     ws=8, ws_syn=4, ax=axs[0])
    # add label and dimension to left-most axis, all other are tied to this one
    ax.set_ylabel('selection coefficient [1/day]', fontsize=fs)
    ax.set_ylim(ymin, ymax)
    add_panel_label(ax, 'B', x_offset=-0.15)

    ax.plot([start,reference.annotation["LTR5'"].location.end],
            ax.get_ylim()[0]*np.ones(2), lw=10, c='k', alpha=0.7)
    ax.text(start, ax.get_ylim()[0]*1.17, "LTR5'", fontsize=fs*0.8, horizontalalignment='left')

    ax.plot([reference.annotation['gag'].location.start, stop],
            ax.get_ylim()[0]*np.ones(2), lw=10, c='k', alpha=0.7)
    ax.text(stop, ax.get_ylim()[0]*1.17, 'gag', fontsize=fs*0.8, horizontalalignment='right')
    ax.set_ylim(ymin, ymax)


    # frame shift region -- no syn fitness cost here since this is in an overlap
    start, stop = 2050, 2150
    feature_names = ['frameshift']
    ax = plot_fitness_costs_along_genome(start, stop, feature_names, data,
                                             minor_af, reference, pheno=None,
                                             ws=8, ws_syn=4, ax=axs[1])

    ax.plot([start, reference.annotation['gag'].location.end],
            ax.get_ylim()[0]*np.ones(2), lw=10, c='k', alpha=0.7)
    ax.text(start, ax.get_ylim()[0]*1.17, 'gag', fontsize=fs*0.8, horizontalalignment='left')

    ax.plot([reference.annotation['pol'].location.start,stop],
            y_second_gene*ax.get_ylim()[0]*np.ones(2), lw=5, c='k', alpha=0.7)
    ax.text(stop, ax.get_ylim()[0]*(y_second_gene+0.17), 'pol', fontsize=fs*0.8, horizontalalignment='right')
    ax.set_xticks([2050, 2150])
    ax.set_ylim(ymin, ymax)

    # plot the cPPT region
    start, stop = 4750, 5000
    feature_names = ['A1','D2', 'cPPT']
    ax = plot_fitness_costs_along_genome(start, stop, feature_names, data,
                                     minor_af, reference, pheno=None,
                                     synnonsyn=synnonsyn['genomewide'],
                                     ws=8, ws_syn=4, ax=axs[2])

    # add label and dimension to left-most axis, all other are tied to this one
    ax.set_ylim(ymin, ymax)

    ax.plot([start,reference.annotation["IN"].location.end],
            ax.get_ylim()[0]*np.ones(2), lw=10, c='k', alpha=0.7)
    ax.text(start, ax.get_ylim()[0]*1.17, "IN", fontsize=fs*0.8, horizontalalignment='left')

    ax.plot([reference.annotation['vif'].location.start, stop],
            y_second_gene*ax.get_ylim()[0]*np.ones(2), lw=10, c='k', alpha=0.7)
    ax.text(stop, ax.get_ylim()[0]*(y_second_gene+0.17), 'vif', fontsize=fs*0.8, horizontalalignment='right')
    ax.set_xticks([4800, 4900])
    ax.set_ylim(ymin, ymax)


    # plot the 3' region
    start, stop = 9050, 9150
    feature_names = ['PPT']
    ax = plot_fitness_costs_along_genome(start, stop, feature_names, data,
                                             minor_af, reference, pheno=None,
                                             synnonsyn=synnonsyn['genomewide'],
                                             ws=8, ws_syn=4, ax=axs[3])

    ax.plot([start, reference.annotation['nef'].location.end],
            ax.get_ylim()[0]*np.ones(2), lw=10, c='k', alpha=0.7)
    ax.text(start, ax.get_ylim()[0]*1.17, 'nef', fontsize=fs*0.8, horizontalalignment='left')

    ax.plot([reference.annotation["LTR3'"].location.start,stop],
            y_second_gene*ax.get_ylim()[0]*np.ones(2), lw=5, c='k', alpha=0.7)
    ax.text(stop, ax.get_ylim()[0]*(y_second_gene+0.17), "LTR3'", fontsize=fs*0.8, horizontalalignment='right')
    ax.set_xticks([9050, 9100,9150])
    ax.set_ylim(ymin, ymax)

    fig.text(0.5, 0.01, 'Position in HIV-1 reference (HXB2) [bp]',
             ha='center',
             fontsize=fs)
    plt.tight_layout(rect=(0, 0.04, 1, 1),w_pad=-1)

    if fname is not None:
        for ext in ['.png', '.svg', '.pdf']:
            plt.savefig(fname+ext)


def add_RNA_properties_to_reference(reference):
    '''
    annotate the reference genome with pairing probabilities from
    siegfried et al and RNA secondary structure predictions by
    Sukosd et al
    '''
    from hivevo.external import load_pairing_probability_NL43
    from hivevo.HIVreference import ReferenceTranslator
    from parse_pairing_probabilities import load_shape
    siegfried = load_pairing_probability_NL43()
    pp = np.zeros(10000)
    pp[siegfried.index] = siegfried.loc[:, 'probability']
    pp[siegfried.loc[:,'partner']] = np.maximum(pp[siegfried.loc[:,'partner']],
                                                siegfried.loc[:,'probability'])

    # siegfried et al pairing probabilities are measured for and NL4-3 sequence
    # translate them to the reference in question (HXB2)
    hxb2_pp = np.zeros(len(reference.seq))
    rt = ReferenceTranslator(ref1 = reference.refname, ref2='NL4-3')
    for i, p in enumerate(pp):
        hxb2_pp[rt.translate(i, 'NL4-3')[1]]=p
    reference.pp = hxb2_pp

    shape = load_shape()
    shape_array = -999.0*np.ones_like(hxb2_pp)
    field = ['1M7 SHAPE MaP', '1M6 SHAPE MaP', 'NMIA SHAPE MaP'][1]
    for pos, val in shape.iterrows():
        r, hxb2pos = rt.translate(pos, 'NL4-3')
        if np.isnan(hxb2pos) or hxb2pos<0:
            continue
        shape_array[hxb2pos] = val[field]
    reference.shape_values = np.ma.array(shape_array, mask=shape_array<-100)

    # load data from Suskod et al NAR.
    suskod = pd.read_csv('../data/Sukosd_etal_NAR_2015.csv')
    fields = suskod.columns[2:]
    suskod_data = {}
    offset = 454
    for field in fields:
        tmp = np.zeros(len(reference.seq))
        for i, p in enumerate(suskod.loc[:,field]):
            tmp[rt.translate(i+offset, 'NL4-3')[1]] = p>0
        suskod_data[field]=tmp
    reference.suskod = suskod_data


def shape_vs_fitness(data, minor_af, shape_data,synnonsyn, ws=100, fname=None, new_fig=True, label=None):
    '''
    calculate the correlation between the pairing probability provided
    by siegfried et al and our estimated selection coefficients
    '''
    if new_fig:
        fig, axs = plt.subplots(2, 1, sharex=True,
                            gridspec_kw={'height_ratios':[6, 1]})
    else:
        fig=plt.gcf()
        axs = fig.get_axes()

    region='genomewide'
    sc = (data['mut_rate'][region]/(af_cutoff+minor_af[region]))
    sc[sc>0.1] = 0.1
    sc[sc<0.001] = 0.001

    ind = ~np.isnan(sc)
    print("overall correlation:", spearmanr(sc[ind], shape_data[ind]))
    ind = (~np.isnan(sc))&(synnonsyn)
    spear = spearmanr(sc[ind], shape_data[ind])
    print("synonymous only correlation:", spear)

    pp_fitness_correlation = []
    for ii in range(len(sc)-ws):
        ind = (~np.isnan(sc[ii:ii+ws]))&synnonsyn[ii:ii+ws]
        cc=0
        if ind.sum()>ws*0.2:
            tmp = spearmanr(shape_data[ii:ii+ws][ind], sc[ii:ii+ws][ind])
            cc = tmp.correlation
        pp_fitness_correlation.append(cc)
    axs[0].plot(np.arange(len(sc)-ws)+ws*0.5, pp_fitness_correlation,
                label=label + r', $\rho='+str(np.round(spear.correlation, 3))+'$')
    axs[0].set_ylabel('rank correlation with fitness costs in '+str(ws)+' base windows')
    if new_fig:
        # add genome annotation
        regs = ['gag', 'pol', 'vif', 'tat','vpu','nef', 'env', 'RRE', "LTR5'", "LTR3'", 'V1', 'V2', 'V3', 'V5']
        annotations = {k: val for k, val in reference.annotation.iteritems() if k in regs}
        annotations = draw_genome(annotations, axs[1])
        axs[1].set_axis_off()
        # vertical lines at feature boundaries
        feats = ['gag', 'pol', 'nef','env', 'RRE', "LTR5'", "LTR3'"]
        vlines = np.unique(annotations.loc[annotations['name'].isin(feats), ['x1', 'x2']])
        for xtmp in vlines:
            axs[0].axvline(xtmp, lw=1, color='0.8')

    if new_fig:
        ngenes = np.zeros(len(reference.seq))
        for feat in reference.annotation.values():
            if feat.type=='gene':
                ngenes[[x for x in feat]] +=1
        overlaps= np.array(ngenes>1.5, dtype=int)
        blocks = zip(np.where(np.diff(overlaps)==1)[0],np.where(np.diff(overlaps)==-1)[0])
        for b in blocks:
            axs[0].plot(b, [0.8, 0.8], c='k', lw=3)
            #axs[0].scatter(np.arange(len(ngenes))[ngenes>1], 0.8*np.ones((ngenes>1).sum()), c='k')

    if new_fig:
        for feat in ['polyA', 'U5', 'U5 stem', 'PBS','cPPT', 'A1','PSI SL1-4']+['RRE'] + ['PPT']+['frameshift']+['TAR', 'TATA', 'SP1', 'TCF-1alpha']:
            #axs[0].text(pos[0][0], 0.74, feat)
            for p in features[feat]:
                axs[0].plot(p, [0.85, 0.85], c='r', lw=3)

    if label is not None:
        axs[0].legend(loc=3)
    if fname is not None:
        for ext in ['.png', '.svg', '.pdf']:
            plt.savefig(fname+ext)


def check_neutrality(minor_af, mut_rates, position_file):
    '''
        analyze the fitness distribution of sites used to calculate the neutral
        mutation rate
    '''
    region='genomewide'
    # make distribution of selection coefficients used to estimate the neutral mutation rate
    ind = (~np.isnan(minor_af['genomewide']))
    slist = mut_rates[region][ind]/(minor_af[region][ind]+af_cutoff)
    s = np.array(slist)
    s[s>=0.1] = 0.1
    s[s<=0.001] = 0.001
    plt.figure()
    plt.hist(s, bins=np.logspace(-3,-1,21), weights=np.ones(len(s))/len(s), alpha=0.5, label='all positions, n='+ str(len(s)))

    neutral_pos = np.array(np.loadtxt(position_file), dtype=int)
    ind = (np.in1d(np.arange(minor_af['genomewide'].shape[0]), neutral_pos))&(~np.isnan(minor_af['genomewide']))
    slist = mut_rates[region][ind]/(minor_af[region][ind]+af_cutoff)
    s = np.array(slist)
    s[s>=0.1] = 0.1
    s[s<=0.001] = 0.001
    plt.hist(s, bins=np.logspace(-3,-1,21), weights=np.ones(len(s))/len(s), alpha=0.5, label='mutation rate positions, n='+ str(len(s)))
    plt.xscale('log')
    plt.legend(loc=9, fontsize=fs*0.8)
    plt.xlabel('fitness cost estimate [1/day]', fontsize=fs)
    plt.ylabel('fraction of sites', fontsize=fs)
    plt.tick_params(labelsize=fs*0.8)
    plt.tight_layout()
    for fmt in ['pdf', 'svg', 'png']:
        plt.savefig('../figures/figure_S2B_mutation_pos_fitness_st_'+args.subtype+'.'+fmt)
    return s


def RNA_correlation_in_genes(data, minor_af, reference, pairings, synnonsyn, fname = 'test.tsv'):
    region='genomewide'
    sc = (data['mut_rate'][region]/(af_cutoff+minor_af[region]))
    sc[sc>0.1] = 0.1
    sc[sc<0.001] = 0.001

    with open(fname, 'w') as ofile:
        for region in ['gag', 'pol','nef',  'env', 'vif']:
            gene_ii = np.in1d(np.arange(len(reference.entropy)), [ii for ii in reference.annotation[region]])
            stmp = sc[gene_ii]
            corr = []
            for pp in pairings:
                pptmp = pp[gene_ii]
                ind = (~np.isnan(stmp))&(synnonsyn[gene_ii])
                corr.append(spearmanr(stmp[ind], pptmp[ind]).correlation)

            ofile.write('\t'.join(map(str, [region] +  [np.round(x,3) for x in corr]))+'\n')
            print(region, corr)


# Script
if __name__=="__main__":

    parser = argparse.ArgumentParser(description='analyze relation of fitness costs and noncoding elements')
    parser.add_argument('--regenerate', action='store_true',
                        help="regenerate data")
    parser.add_argument('--subtype', choices=['B', 'any'], default='B',
                        help='subtype to compare against')
    args = parser.parse_args()

    # NOTE: HXB2 alignment has way more sequences resulting in better correlations
    reference = HIVreference(refname='HXB2', subtype=args.subtype)
    genes = ['gag', 'nef', 'env', 'vif','pol', 'vpr', 'vpu']
    # Intermediate data are saved to file for faster access later on
    fn = '../data/fitness_pooled_noncoding/avg_noncoding_allele_frequency_st_'+args.subtype+'.pickle.gz'
    if not os.path.isfile(fn) or args.regenerate:
        if args.subtype=='B':
            patient_codes = ['p2','p3', 'p5','p7', 'p8', 'p9','p10', 'p11'] # subtype B only
        else:
            patient_codes = ['p1', 'p2','p3','p5','p6','p7', 'p8', 'p9','p10', 'p11'] # all subtypes

        # gag and nef are loaded since they overlap with relevnat non-coding structures
        # and we need to know which positions have synonymous mutations
        data = collect_data(patient_codes,genes, reference, synnonsyn=True)
        tmp_data = collect_data(patient_codes, ['genomewide'], reference, synnonsyn=False)
        for k in data:
            data[k].update(tmp_data[k])

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
    av = process_average_allele_frequencies(data, genes, nbootstraps=0,
                                            synnonsyn=True)
    combined_af = av['combined_af']
    combined_entropy = av['combined_entropy']
    minor_af = av['minor_af']
    synnonsyn = av['synnonsyn']
    synnonsyn_unconstrained = av['synnonsyn_unconstrained']
    av = process_average_allele_frequencies(data, ['genomewide'], nbootstraps=0,
                                            synnonsyn=False)
    combined_af.update(av['combined_af'])
    combined_entropy.update(av['combined_entropy'])
    minor_af.update(av['minor_af'])
    synnonsyn['genomewide'] = np.ones_like(minor_af['genomewide'], dtype=bool)
    synnonsyn_unconstrained['genomewide'] = np.ones_like(minor_af['genomewide'], dtype=bool)
    for gene in genes:
        pos = [x for x in reference.annotation[gene]]
        synnonsyn_unconstrained['genomewide'][pos] = synnonsyn_unconstrained[gene]
        synnonsyn['genomewide'][pos] = synnonsyn[gene]

    plot_non_coding_figure(data, minor_af, synnonsyn_unconstrained, reference,
                           fname='../figures/figure_4B_st_'+args.subtype)

    # Check SHAPE vs fitness
    add_RNA_properties_to_reference(reference)
    ws=100
    subset_of_positions = synnonsyn_unconstrained['genomewide']
    #subset_of_positions = np.ones_like(synnonsyn_unconstrained['genomewide'], dtype=bool)
    shape_vs_fitness(data, minor_af, -reference.entropy, subset_of_positions, ws=ws,
                     fname=None, new_fig=True,
                     label=("group M" if args.subtype=='any' else 'subtype B')+ "diversity")

#    shape_vs_fitness(data, minor_af, -reference.shape_values, subset_of_positions, ws=ws,
#                     fname=None, new_fig=False, label="SHAPE")

    shape_vs_fitness(data, minor_af, reference.pp, subset_of_positions, ws=ws,
                     fname=None, new_fig=False, label="pairing probability, Siegfried et al.")

    for k in reference.suskod:
        if k.startswith('PPfold, SHAPE'):
            print(k)
            shape_vs_fitness(data, minor_af, reference.suskod[k], subset_of_positions, ws=ws,
                     fname='../figures/pairing_fitness_correlation_st_'+args.subtype+'_ws_'+str(ws),
                     new_fig=False, label=k.replace(', ','+')+'; from Sukosd et al.')

    # check the neutrality of the positions used to determine the neutral mutation rate.
    s = check_neutrality(minor_af, data['mut_rate'], '../data/mutation_rates/mutation_rate_positions_0.3_gp120.txt')

    pairings = [reference.pp, reference.suskod['PPfold, SHAPE']]
    RNA_correlation_in_genes(data, minor_af, reference, pairings, subset_of_positions,
                             fname='../data/RNA_phenotype_correlations_st_'+args.subtype+'.tsv')

