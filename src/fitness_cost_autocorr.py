# vim: fdm=indent
'''
author:     Fabio Zanini
date:       07/10/15
content:    Estimate fitness costs by autocorrelation.
'''
# Modules
import os
import sys
import argparse
from itertools import izip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Bio.Seq import translate

from hivevo.patients import Patient
from hivevo.HIVreference import HIVreference
from hivevo.sequence import alpha, alphal


# Globals
patients = ['p1', 'p2', 'p3','p5', 'p6', 'p8', 'p9', 'p11']



# Functions
def add_binned_column(df, bins, to_bin):
    # FIXME: this works, but is a little cryptic
    df.loc[:, to_bin+'_bin'] = np.minimum(len(bins)-2,
                                          np.maximum(0,np.searchsorted(bins, df.loc[:,to_bin])-1))



# Script
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fitness cost via autocorrelation')
    parser.add_argument('--regenerate', action='store_true',
                        help="regenerate data")
    args = parser.parse_args()

    if args.regenerate:
        cov_min = 100
        from fitness_cost_saturation import collect_data
        data = collect_data(patients, cov_min=cov_min)
        data.to_pickle('../data/fitness_cost_data.pickle')
    else:
        data = pd.read_pickle('../data/fitness_cost_data.pickle')

    t_bins = np.array([0, 500, 1000, 1500, 2000, 3000], int)
    t_binc = 0.5 * (t_bins[:-1] + t_bins[1:])
    add_binned_column(data, t_bins, 'time')
    data['time_binc'] = t_binc[data['time_bin']]

    S_bins = np.percentile(data['S'], np.linspace(0, 100, 8))
    S_binc = 0.5 * (S_bins[:-1] + S_bins[1:])
    add_binned_column(data, S_bins, 'S')
    data['S_binc'] = S_binc[data['S_bin']]

    # Focus on conserved sites
    dcons = (data
             .groupby(['pos_ref', 'time_binc'])
             .filter(lambda x: len(set(x['pcode'])) == len(patients)))

    # Average frequencies across patients
    dav = (dcons
           .loc[:, ['pos_ref', 'time_binc', 'S_binc', 'mu', 'af']]
           .groupby(['pos_ref', 'time_binc'])
           .mean())
