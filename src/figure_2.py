# vim: fdm=indent
'''
author:     Fabio Zanini
date:       15/06/15
content:    Make figure for the fitness cost estimate from the saturation curves.
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

from util import add_binned_column, boot_strap_patients



# Functions
def load_data_saturation():
    import cPickle as pickle
    fn = 'data/fitness_cost_saturation_plot.pickle'
    with open(fn, 'r') as f:
        data = pickle.load(f)
    return data


def load_data_KL():
    '''Load data from Vadim's KL approach'''
    S_quantiles = np.loadtxt('data/smuD_KL_quantiles.txt')
    S_center = 0.5 * (S_quantiles[:-1] + S_quantiles[1:])

    raw = np.loadtxt('data/smuD_KL.txt')
    s = raw[:, :-2]
    
    # Geometric means, but linear is the same
    s_mean = np.exp(np.log(s).mean(axis=0))
    s_std = s_mean * np.log(s).std(axis=0)

    data = pd.DataFrame({'mean': s_mean, 'std': s_std}, index=S_center)
    data.index.name = 'Entropy'
    data.name = 'Fitness costs'

    return data


def plot_fit(data_sat, data_KL):
    from matplotlib import cm
    fig_width = 5
    fs = 16
    fig, axs = plt.subplots(1, 2,
                            figsize=(2 * fig_width, fig_width))


    data_to_fit = data_sat['data_to_fit']
    mu = data_sat['mu']
    s = data_sat['s']

    fun = lambda x, s: mu / s * (1.0 - np.exp(-s * x))

    # PANEL A: data and fits
    ax = axs[0]
    for iS, (S, datum) in enumerate(data_to_fit.iterrows()):
        x = np.array(datum.index)
        y = np.array(datum)
        color = cm.jet(1.0 * iS / data_to_fit.shape[0])

        # Most conserved group is dashed
        if iS == 0:
            ls = '--'
        else:
            ls = '-'

        ax.scatter(x, y,
                   s=70,
                   color=color,
                  )

        xfit = np.linspace(0, 3000)
        yfit = fun(xfit, s.loc[S, 's'])
        ax.plot(xfit, yfit,
                lw=2,
                color=color,
                ls=ls,
               )

    ax.set_xlabel('days since EDI', fontsize=fs)
    ax.set_ylabel('Average allele frequency', fontsize=fs)
    ax.set_xlim(-200, 3200)
    ax.set_ylim(-0.0005, 0.025)
    ax.set_xticks(np.linspace(0, 0.005, 5))
    ax.set_xticks([0, 1000, 2000, 3000])
    ax.xaxis.set_tick_params(labelsize=fs)
    ax.yaxis.set_tick_params(labelsize=fs)

    ax.text(0, 0.023,
            r'$\mu = 1.2 \cdot 10^{-5}$ per day',
            fontsize=16)
    ax.plot([200, 1300], [0.007, 0.007 + (1300 - 200) * mu], lw=1.5, c='k')

    # PANEL B: costs
    ax = axs[1]

    # B1: Saturation fit
    x = np.array(s.index)
    y = np.array(s['s'])
    dy = np.array(s['ds'])

    ymin = 0.1

    x = x[1:]
    y = y[1:]
    dy = dy[1:]

    ax.errorbar(x, y,
                yerr=dy,
                lw=2,
                color='b',
                label='Sat fit',
               )

    ax.plot([1e-3, s.index[1]],
            [s['s'].iloc[0], s['s'].iloc[1]],
            lw=2,
            ls='--',
            color='b',
           )
    ax.errorbar([1e-3], [s['s'].iloc[0]],
                yerr=[s['ds'].iloc[0]],
                lw=2,
                color='b'
               )

    # Arrow for the most conserved quantile
    if False:
        ax.annotate('Full conservation',
                    xy=(1.1e-3, 0.9 * s['s'].iloc[0]),
                    xytext=(1.1e-3, 0.01 * s['s'].iloc[0]),
                    arrowprops={'facecolor': 'black',
                                'width': 1.5,
                                'headlength': 10,
                                'shrink': 0.1,
                               },
                    ha='left',
                    va='center',
                    fontsize=fs,
                   )


    # Bw: KL fit
    x = np.array(data_KL.index)
    y = np.array(data_KL['mean'])
    dy = np.array(data_KL['std'])
    ax.errorbar(x, y, yerr=dy,
                lw=2,
                color='darkred',
                label='KL fit',
               )

    ax.legend(loc='upper right', fontsize=16)
    ax.set_xlabel('Variability in group M [bits]', fontsize=fs)
    ax.set_ylabel('Fitness cost', fontsize=fs)
    ax.set_xlim(0.9e-3, 2.5)
    ax.set_ylim(9e-5, 0.11)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_tick_params(labelsize=fs)
    ax.yaxis.set_tick_params(labelsize=fs)

    plt.tight_layout()
    plt.ion()
    plt.show()



# Script
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Figure 2')
    args = parser.parse_args()

    data_sat = load_data_saturation()
    data_KL = load_data_KL()

    plot_fit(data_sat, data_KL)


