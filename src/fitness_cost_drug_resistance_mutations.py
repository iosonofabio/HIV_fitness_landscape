# vim: fdm=indent
'''
author:     Richard Neher
date:       21/03/16
content:    Plot the cost of drug resistance mutations. NOTE: this script can be
            run only after combined_af_aa.py, which stores the analyzed data.
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
from fitness_pooled import process_average_allele_frequencies, draw_genome, af_average, get_final_state, load_mutation_rates
from fitness_pooled_aa import calc_amino_acid_mutation_rates



# Globals
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
                   },
             #http://hivdb.stanford.edu/pages/download/resistanceMutations_handout.pdf
             # offset includes RT and p15 (RNase H)
             'INI': {'offset': 56 + 99 + 560 - 1,
                     'mutations': [('T', 66, 'IAK'),
                                   ('E', 92, 'Q'),
                                   ('E', 138, 'KA'),
                                   # NOTE: G140SAC has a very low neutral mut rate, so we are unable to calculate the cost faithfully
                                   #('G', 140, 'SAC'),
                                   ('Y', 143, 'CRH'),
                                   ('S', 147, 'G'),
                                   ('Q', 148, 'HRK'),
                                   ('N', 155, 'H')]},
            }



# Functions
def plot_drug_resistance_mutations(data, aa_mutation_rates, fname=None):
    '''Plot the frequency of drug resistance mutations'''
    import matplotlib.patches as patches

    fs = 16
    region = 'pol'
    pcodes = data['init_codon'][region].keys()

    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 6]})
    ax = axs[1]

    drug_afs_items = []
    mut_types = []
    drug_classes = ['PI', 'NRTI', 'NNRTI', 'INI']
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
    dd = afdr.iloc[[0, 1, 2, 3, 4]].copy()
    dd.index = ['x', 'freq', 'size', 'cost', 'mr']
    dd.loc['x'] = np.arange(dd.shape[1])
    dd.loc['freq'] = 2e-5
    dd.loc['n'] = afdr.shape[0] - (afdr > 1e-4).sum(axis=0)
    dd.loc['size'] = dd.loc['n']**(1.4) + 13
    dd.loc['cost'] = 1.0 / afdr.fillna(0).mean(axis=0)
    dd.loc['mr'] = 0
    # NOTE: the first 6 mutations are in PR, the rest in RT
    import re
    from Bio.Seq import translate
    reference = HIVreference(refname='HXB2', load_alignment=False)
    seq_PR = reference.annotation['PR'].extract(reference.seq)
    seq_RT = reference.annotation['RT'].extract(reference.seq)
    seq_IN = reference.annotation['IN'].extract(reference.seq)
    murate = load_mutation_rates()['mu']
    for i, mut in enumerate(dd.T.index):
        mr = 0
        if i < 6:
            seq_tmp = seq_PR
        elif i < 6 + 5 + 4:
            seq_tmp = seq_RT
        else:
            seq_tmp = seq_IN
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
        dd.loc['mr', mut] = mr

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
    plt.xticks(rotation=50)
    plt.ylabel('minor variant frequency', fontsize=fs)
    plt.tick_params(labelsize=fs*0.8)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_horizontalalignment('right')

    # Fitness cost at the top
    ax1 = axs[0]
    ax1.set_xlim(*ax.get_xlim())
    ax1.set_xticks(ax.get_xticks() + 0.5)
    ax1.set_xticklabels([])
    ax1.set_ylim(1e-3, 1)
    ax1.set_yticks([1e-3, 1e-2, 1e-1, 1])
    ax1.yaxis.set_tick_params(labelsize=fs*0.8)
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

    parser = argparse.ArgumentParser(description='fitness costs of drug resistance mutations')
    parser.add_argument('--subtype', choices=['B', 'any'], default='any',
                        help='subtype to compare against')
    args = parser.parse_args()

    fn = '../data/fitness_pooled_aa/avg_aa_allele_frequency_st_'+args.subtype+'.pickle.gz'

    if not os.path.isfile(fn):
        raise IOError('Data file not found. Please run combined_af_aa.py first')
    else:
        with gzip.open(fn) as ifile:
            data = cPickle.load(ifile)

    aa_mutation_rates, total_nonsyn_mutation_rates = calc_amino_acid_mutation_rates()
    plot_drug_resistance_mutations(data, aa_mutation_rates, '../figures/figure_6_subtype_'+args.subtype+'_withcost')
