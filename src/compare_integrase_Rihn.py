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

    # I aligned to HXB2, there are no indels
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
    return df[['mut', 'cost']].set_index('mut')



# Script
if __name__ == '__main__':

    seq = get_integrase_Rihn()
    costs = load_costs_Rihn()
