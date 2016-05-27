# vim: fdm=indent
'''
author:     Fabio Zanini
date:       26/05/16
content:    Compare our results in integrase with Rihn et al 2015.
'''
# Modules
from __future__ import division, print_function

import os
import sys
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import translate
import matplotlib.pyplot as plt
import seaborn as sns

from seqanpy import align_overlap

from hivevo.patients import Patient
from hivevo.HIVreference import HIVreference


# Functions
def get_plasmid_Rihn():
    '''Get the plasmid sequence from the Rihn paper (15kb)'''
    fn = 'data/Rihn_et_al_2015_pNHG-CapNM.gb'
    return SeqIO.read(fn, 'gb')


def get_integrase_Rihn():
    '''Get the integrase sequence from Rihn'''
    seq = get_plasmid_Rihn()

    # I aligned to HXB2 or NL4-3, there are no indels
    # That is no surprise as their plasmid is an HXB2/NL4-3 recombinant
    #reference = HIVreference(refname='HXB2', subtype='B', load_alignment=False)
    #ref_in = reference.annotation['IN'].extract(reference.seq)

    #(score, ali1, ali2) = align_overlap(seq, ref_in)
    #start = len(ali2) - len(ali2.lstrip('-'))
    #end = len(ali2.rstrip('-'))
    #seq_in = ali1[start: end]

    seq_in = seq[4229: 5096]
    seq_in.description = 'Cloning vector pNHG-CapNM, integrase sequence'
    return seq_in


def load_costs_Rihn():
    '''Load the replication costs from Rihn'''
    fn = 'data/Rihn_et_al_2015_Tables1-2.csv'
    df =  (pd.read_csv(fn, sep='\t')
           .rename(columns={'# Mutation': 'mut', 'replication': 'rep'})
           .loc[:, ['mut', 'rep']]
           )
    df['rep'] /= 100.
    df['cost'] = np.maximum(0, 1.0 - df['rep'])
    df['pos'] = [int(it[1:-1]) for it in df['mut']]
    df['NL4-3'] = [it[0] for it in df['mut']]
    df['nuc'] = [it[-1] for it in df['mut']]
    return df[['pos', 'NL4-3', 'nuc', 'mut', 'cost']].sort_values('pos')


def load_costs_ours():
    '''Load the fitness costs from us'''
    fn = 'data/pol_selection_coefficients_st_B.tsv'
    df = (pd.read_csv(fn, sep='\t', header=1)
          .rename(columns={'# position': 'pos'}))
    ref = ''.join(df['NL4-3'])

    # Cut out only the integrase
    ref = ref[715: 715 + 289]
    df = df.iloc[715: 715 + 289]
    df.index -= 715
    df.pos -= 715

    return ref, df


def find_Rihn_multiple_alleles(costs):
    '''Find the positions for which Rihn has multiple alleles'''
    d = {key: value for key, value in Counter(costs_Rihn['pos']).iteritems() if value >= 2}
    return costs.loc[costs['pos'].isin(d.keys())]


def get_our_costs_at_Rihn(costs_Rihn, costs_ours):
    '''Get our costs at their positions'''
    c = costs_ours.set_index('pos').loc[costs_Rihn['pos']]['median']
    c_float = []
    for ci in c:
        if ci == '<0.001':
            ci = 0
        elif ci == '>0.1':
            ci = 1
        else:
            ci = float(ci)
        c_float.append(ci)
    c[:] = c_float

    comp = (pd.concat([c, costs_Rihn.set_index('pos')['cost']], axis=1)
            .rename(columns={'median': 'ours', 'cost': 'Rihn'}))

    return comp


def plot_comparison(comp):
    '''Plot comparison of us with Rihn'''
    fig, ax = plt.subplots(figsize=(5, 5))

    from scipy.stats import spearmanr
    s = spearmanr(comp['ours'], comp['Rihn'])
    print(s)

    shuffled_Rihn = np.array(comp['Rihn'])
    for i in xrange(5):
        np.random.shuffle(shuffled_Rihn)
        s_shuffled = spearmanr(np.array(comp['ours']), shuffled_Rihn)
        print('Shuffle, try', i, s_shuffled)

    label = 'rho = '+str(s[0])[:4]+'\n'+'P = '+str(s[1])[:5]
    ax.scatter(comp['ours'] + 0.0001, comp['Rihn'] + 0.0001,
               s=40, color='k',
               label=label)
    ax.set_xlabel('Our cost')
    ax.set_ylabel('Cost from Rihn et al. 2015')

    ax.set_xlim(0.0001, 1.01)
    ax.set_ylim(0.0001, 1.01)
    ax.set_xscale('log')
    ax.text(1.2e-4, 0.7, label, fontsize=12)

    plt.tight_layout()
    plt.ion()
    plt.show()

    return fig



# Script
if __name__ == '__main__':

    seq = get_integrase_Rihn()
    costs_Rihn = load_costs_Rihn()
    ref, costs_ours = load_costs_ours()

    comp = get_our_costs_at_Rihn(costs_Rihn, costs_ours)

    fig = plot_comparison(comp)
    for ext in ['svg', 'pdf', 'png']:
        fig.savefig('figures/comparison_integrase_Rihn2015.'+ext)
