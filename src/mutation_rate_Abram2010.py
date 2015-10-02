# vim: fdm=marker
'''
author:     Fabio Zanini
date:       09/01/15
content:    Study divergence at conserved/non- synonymous sites in different
            patients.
'''
# Modules
import os
import argparse
from itertools import izip
import numpy as np
import pandas as pd

from hivevo.sequence import alpha



# Functions
def get_mu_Abram2010(normalize=True, strand='both'):
    '''Get the mutation rate matrix from Abram 2010'''
    muts = [a+'->'+b for a in alpha[:4] for b in alpha[:4] if a != b]

    muAbram = pd.Series(np.zeros(len(muts)), index=muts, dtype=float)
    muAbram.name = 'mutation rate Abram 2010'

    if strand in ['fwd', 'both']:
        muAbram['C->A'] += 14
        muAbram['G->A'] += 146
        muAbram['T->A'] += 20
        muAbram['A->C'] += 1
        muAbram['G->C'] += 2
        muAbram['T->C'] += 18
        muAbram['A->G'] += 29
        muAbram['C->G'] += 0
        muAbram['T->G'] += 6
        muAbram['A->T'] += 3
        muAbram['C->T'] += 81
        muAbram['G->T'] += 4

    if strand in ['rev', 'both']:
        muAbram['C->A'] += 24
        muAbram['G->A'] += 113
        muAbram['T->A'] += 32
        muAbram['A->C'] += 1
        muAbram['G->C'] += 2
        muAbram['T->C'] += 25
        muAbram['A->G'] += 13
        muAbram['C->G'] += 1
        muAbram['T->G'] += 8
        muAbram['A->T'] += 0
        muAbram['C->T'] += 61
        muAbram['G->T'] += 0

    if normalize:
        muAbramAv = 1.3e-5
        muAbram *= muAbramAv / (muAbram.sum() / 4.0)

    return muAbram



# Script
if __name__ == '__main__':

    # Parse input args
    parser = argparse.ArgumentParser(
        description='Get mutation rate from Abram2010',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbosity level [0-4]')

    args = parser.parse_args()
    VERBOSE = args.verbose

    muA = get_mu_Abram2010()
