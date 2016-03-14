# vim: fdm=indent
'''
author:     Fabio Zanini
date:       15/06/15
content:    Make figure for the mutation rate.
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
from collections import defaultdict
from Bio.Seq import translate

from hivevo.patients import Patient
from hivevo.HIVreference import HIVreference
from hivevo.sequence import alpha, alphal

from util import add_binned_column, boot_strap_patients

fs=18

# Functions
def load_mutation_rate():
    fn = '../data/mutation_rate.pickle'
    mu =  pd.read_pickle(fn)
    return mu


def plot_comparison(mu, muA, dmulog10=None, dmuAlog10=None):
    '''Compare new estimate for mu with Abram et al 2010'''
    xmin = -7.3
    xmax = -3.9
    fs = 16

    fig, ax = plt.subplots()

    x = []
    y = []
    for key in mu.index:
        x.append(np.log10(mu[key]))
        y.append(np.log10(muA[key]))

    if dmulog10 is not None:
        dx = [dmulog10[key] for key in mu.index]
    else:
        dx = None

    if dmuAlog10 is not None:
        dy = [dmuAlog10[key] for key in mu.index]
    else:
        dy = None

    from scipy.stats import pearsonr, spearmanr
    R = pearsonr(x, y)[0]
    rho = spearmanr(x, y)[0]

    label = (r'Pearson $r = {0:3.0%}$'.format(np.round(R, 2))+
             '\n'+
             r'Spearman $\rho = {0:3.0%}$'.format(np.round(rho, 2)))
    label = label.replace('%','\%')

    ax.errorbar(x, y,
                xerr=dx, yerr=dy,
                ls='none',
                ms=10,
                marker='o',
                label=label)

    ax.plot(np.linspace(xmin, xmax, 1000),
            np.linspace(xmin, xmax, 1000),
            color='grey',
            lw=1,
            alpha=0.7)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_xlabel(r'rate [1 / day / site]', fontsize=fs)
    ax.set_ylabel(r'rate (Abram et al.2010)', fontsize=fs)
    ax.set_xticks([-7, -6, -5, -4])
    ax.set_yticks([-7, -6, -5, -4])
    ax.set_xticklabels(['$10^{-7}$', '$10^{-6}$',
                        '$10^{-5}$', '$10^{-4}$'])
    ax.set_yticklabels(['$10^{-7}$', '$10^{-6}$',
                        '$10^{-5}$', '$10^{-4}$'])
    ax.xaxis.set_tick_params(labelsize=fs)
    ax.yaxis.set_tick_params(labelsize=fs)
    ax.text(0.04, 0.93, label,
            va='top',
            transform=ax.transAxes,
            fontsize=fs)
    ax.grid(True)

    plt.tight_layout()

    plt.ion()
    plt.show()

    for ext in ['svg', 'png', 'pdf']:
        fig.savefig('../figures/figure_S1.'+ext)

    return ax



# Script
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Figure S1')
    args = parser.parse_args()

    mu = load_mutation_rate()

    plot_comparison(mu['mu'],
                    mu['muA'],
                    dmulog10=mu['dmulog10'],
                    dmuAlog10=mu['dmuAlog10'])
