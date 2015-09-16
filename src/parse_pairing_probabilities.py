# vim: fdm=indent
'''
author:     Fabio Zanini
date:       16/09/15
content:    Parse pairing probabilities from Siegfried et al. 2014 (SHAPE-MaP
            of NL4-3).
'''
# Modules
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm



# Globals
def load_pairing_probability():
    data_filename = 'data/nmeth.3029-S4.txt'
    data = pd.read_csv(data_filename, header=1, sep='\t', index_col=0)
    data.rename(columns={'j': 'partner'}, inplace=True)
    data.index.name = 'position'

    # Use positive pairing probabilities
    data.rename(columns={'-log10(Probability)': 'probability'}, inplace=True)
    data['probability'] = 10**(-data['probability'])

    # Indices start from 0 in Python
    data.index -= 1
    data['partner'] -= 1

    # The NL4-3 sequence starts later
    start = 454
    data.index += start
    data['partner'] += start

    return data


def load_shape():
    data_filename = 'data/nmeth.3029-S2.csv'
    data = pd.read_csv(data_filename, sep='\t', index_col=0)
    data.index.name = 'position'

    # Indices start from 0 in Python
    data.index -= 1

    # The NL4-3 sequence starts later
    start = 454
    data.index += start

    return data



# Script
if __name__ == '__main__':

    pp = load_pairing_probability()

    shape = load_shape()


    from hivevo.HIVreference import HIVreference
    ref = HIVreference('NL4-3', load_alignment=False)
    seq = ref.seq.seq

    pairings = {'A': 'T', 'C': 'G', 'T': 'A', 'G': 'C'}
    for position, (partner, prob) in pp.iterrows():
        partner = int(partner)
        if ((seq[partner] != pairings[seq[position]]) and 
            # Wobble pairs
            (frozenset([seq[position], seq[partner]]) != frozenset(['G', 'T']))):

            print position, seq[position], seq[partner], pairings[seq[position]]
         

    #fig, ax = plt.subplots()
    #ax.hist(shape['probability'], bins=np.linspace(0, 1, 10))
    #ax.set_xlabel('Pairing probability')
    #ax.set_ylabel('Number of sites')

    #plt.ion()
    #plt.show()
