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
    return pd.read_pickle(fn)['mu']


def plot_mutation_rate(mu, ax=None):
    '''Plot accumulation of mutations and fits'''
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(13, 8))
    else:
        fig=None
    lim = 6.7
    ax.set_xlim(-lim, lim * 2 + 10)
    ax.set_ylim(lim, -lim)
    ax.axis('off')

    nucs = ['A', 'C', 'G', 'T']
    rc = 5
    r = 1.5
    for iy, yc in enumerate([-rc, rc]):
        for ix, xc in enumerate([-rc, rc]):
            circ = plt.Circle((xc, yc), radius=r,
                              edgecolor='black',
                              facecolor=([0.9] * 3),
                              lw=2.5,
                             )
            ax.add_patch(circ)
            i = 2 * iy + ix
            ax.text(xc, yc, nucs[i], ha='center', va='center', fontsize=34)

    def get_arrow_properties(mut, scale=1.0):
        from matplotlib import cm
        cmap = cm.jet
        wmin = 0.2
        wmax = 0.6
        fun = lambda x: np.log10(x)
        mumin = fun(1e-7)
        mumax = fun(2e-5)
        if isinstance(mut, basestring):
            m = fun(mu.loc[mut])
        else:
            m = fun(mut)
        frac = (m - mumin) / (mumax - mumin)
        w = wmin + frac * (wmax - wmin)
        return {'width': scale * w,
                'head_width': scale * w * 2.5,
                'facecolor': cmap(1.0 * frac),
                'edgecolor': cmap(1.0 * frac),
               }

    gap = 0.5
    ax.arrow(-(rc + gap), -(rc - r - 0.2), 0, 2 * (rc - r - 0.2), length_includes_head=True, **get_arrow_properties('A->G'))
    ax.arrow(-(rc - gap), (rc - r - 0.2), 0, -2 * (rc - r - 0.2), length_includes_head=True, **get_arrow_properties('G->A'))
    ax.arrow(+(rc - gap), -(rc - r - 0.2), 0, 2 * (rc - r - 0.2), length_includes_head=True, **get_arrow_properties('C->T'))
    ax.arrow(+(rc + gap), +(rc - r - 0.2), 0, -2 * (rc - r - 0.2), length_includes_head=True, **get_arrow_properties('T->C'))
    ax.arrow(-(rc - r - 0.2), -(rc + gap), 2 * (rc - r - 0.2), 0, length_includes_head=True, **get_arrow_properties('A->C'))
    ax.arrow(+(rc - r - 0.2), -(rc - gap), -2 * (rc - r - 0.2), 0, length_includes_head=True, **get_arrow_properties('C->A'))
    ax.arrow(-(rc - r - 0.2), +(rc - gap), 2 * (rc - r - 0.2), 0, length_includes_head=True, **get_arrow_properties('G->T'))
    ax.arrow(+(rc - r - 0.2), +(rc + gap), -2 * (rc - r - 0.2), 0, length_includes_head=True, **get_arrow_properties('T->G'))
    ax.arrow(-(rc - r - 0.7), -(rc - r - 0.2), 2 * (rc - r - 0.4), 2 * (rc - r - 0.4), length_includes_head=True, **get_arrow_properties('A->T'))
    ax.arrow(+(rc - r - 0.7), +(rc - r - 0.2), -2 * (rc - r - 0.4), -2 * (rc - r - 0.4), length_includes_head=True, **get_arrow_properties('T->A'))
    ax.arrow(-(rc - r - 0.5), +(rc - r - 0.2), 2 * (rc - r - 0.3), -2 * (rc - r - 0.3), length_includes_head=True, **get_arrow_properties('G->C'))
    ax.arrow(+(rc - r - 0.5), -(rc - r - 0.2), -2 * (rc - r - 0.3), +2 * (rc - r - 0.3), length_includes_head=True, **get_arrow_properties('C->G'))


    oft = 0.8
    ax.text(rc + 2.4, - rc + oft - 1.7, 'Rates [per site per day]:', fontsize=fs*1.2)

    def write_mut(mut, dy):
        decade = int(np.floor(np.log10(mu.loc[mut])))
        flo = mu.loc[mut] * 10**(-decade)
        mtxt = '$' + '{:1.1f}'.format(flo) + ' \cdot ' + '10^{'+str(decade)+'}$'
        ax.text(rc + 2.3, -rc + oft - 0.6 + dy, mut[0]+u' \u2192 '+mut[-1], fontsize=fs)
        ax.text(rc + 10.2, -rc + oft - 0.6 + dy, mtxt, ha='right', fontsize=fs)
        ax.arrow(rc + 4.9, - rc + oft - 0.8 + dy, 2.0, 0, length_includes_head=True,
                 **get_arrow_properties(mu.loc[mut], scale=0.7))

    dy = 0
    muts = ['G->C', 'A->T', 'C->G', 'A->C', 'G->T', 'T->A',
            'T->G', 'C->A', 'A->G', 'T->C', 'C->T', 'G->A']
    for mut in muts:
        write_mut(mut, dy)
        dy += 1

    #ax.arrow(rc + 3, - rc + oft, rc, 0, length_includes_head=True, **get_arrow_properties(1e-7))
    #ax.text(rc + 3 + rc + 3.5, - rc + oft, r'$10^{-7}$', ha='right', va='center', fontsize=34)

    #ax.arrow(rc + 3, - rc + oft + 2, rc, 0, length_includes_head=True, **get_arrow_properties(3e-7))
    #ax.text(rc + 3 + rc + 3.5, -rc + oft + 2, r'$3 \cdot 10^{-7}$', ha='right', va='center', fontsize=34)

    #ax.arrow(rc + 3, - rc + oft + 4, rc, 0, length_includes_head=True, **get_arrow_properties(1e-6))
    #ax.text(rc + 3 + rc + 3.5, -rc + oft + 4, r'$10^{-6}$', ha='right', va='center', fontsize=34)

    #ax.arrow(rc + 3, - rc + oft + 6, rc, 0, length_includes_head=True, **get_arrow_properties(3e-6))
    #ax.text(rc + 3 + rc + 3.5, -rc + oft + 6, r'$3 \cdot 10^{-6}$', ha='right', va='center', fontsize=34)

    #ax.arrow(rc + 3, - rc + oft + 8, rc, 0, length_includes_head=True, **get_arrow_properties(1e-5))
    #ax.text(rc + 3 + rc + 3.5, -rc + oft + 8, r'$10^{-5}$', ha='right', va='center', fontsize=34)

    #ax.arrow(rc + 3, - rc + oft + 10, rc, 0, length_includes_head=True, **get_arrow_properties(2e-5))
    #ax.text(rc + 3 + rc + 3.5, -rc + oft + 10, r'$2 \cdot 10^{-5}$', ha='right', va='center', fontsize=34)

    plt.tight_layout()

    plt.ion()
    plt.show()

    return fig, ax


# Script
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Mutation rate')
    #parser.add_argument('--regenerate', action='store_true',
    #                    help="regenerate data")
    args = parser.parse_args()

    mu = load_mutation_rate()
    fig, ax = plot_mutation_rate(mu)
