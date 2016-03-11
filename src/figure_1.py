# vim: fdm=indent
'''
author:     Fabio Zanini
date:       15/06/15
content:    Make figure for the mutation rate.
'''
# Modules
from __future__ import print_function
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



# Functions
def get_mu_Abram2010(normalize=True, strand='both', with_std=False):
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

    muAbramAv = 1.3e-5
    # Assuming an even composition of the lacZ substrate
    nSitesPerNucleotide = 0.25 * muAbram.sum() / muAbramAv

    if normalize:
        muAbram1 = muAbram / nSitesPerNucleotide
    else:
        muAbram1 = muAbram

    if not with_std:
        return muAbram1
    else:
        std = np.sqrt(muAbram * (1 - muAbram / nSitesPerNucleotide))
        if normalize:
            std /= nSitesPerNucleotide

        return {'mu': muAbram1,
                'std': std}


def get_mutation_matrix(data):
    '''Calculate the mutation rate matrix'''
    def get_mu(data):
        d = (data
             .loc[:, ['af', 'time_binc', 'mut']]
             .groupby(['mut', 'time_binc'])
             .mean()
             .unstack('time_binc')
             .loc[:, 'af'])

        rates = {}
        for mut, aft in d.iterrows():
            times = np.array(aft.index)
            aft = np.array(aft)
            rate = np.inner(aft, times) / np.inner(times, times)
            rates[mut] = rate

        mu = pd.Series(rates)
        mu.name = 'mutation rate from longitudinal data'

        return mu

    mu = get_mu(data)

    # Bootstrap
    dmulog10 = mu.copy()
    muBS = boot_strap_patients(data, get_mu, n_bootstrap=100)
    for key, _ in dmulog10.iteritems():
        dmulog10[key] = np.std([np.log10(tmp[key]) for tmp in muBS])

    return mu, dmulog10


def plot_mutation_increase(data, mu=None, axs=None):
    '''Plot accumulation of mutations and fits'''
    cmap = sns.color_palette()
    transitions =  ['A->G','C->T', 'G->A', 'T->C']
    transversions_pair = ['A->T', 'C->G',  'G->C', 'T->A']
    transversions_np = ['A->C', 'C->A', 'G->T', 'T->G']

    d = (data
         .loc[:, ['af', 'time_binc', 'mut']]
         .groupby(['mut', 'time_binc'])
         .mean()
         .unstack('time_binc')
         .loc[:, 'af'])

    dsamples = (data
         .loc[:, ['af', 'time', 'time_binc', 'mut']]
         .groupby(['mut', 'time'])
         .mean())
    sampleavg = {}
    for mut, aft in dsamples.iterrows():
        if mut[0] not in sampleavg:
            sampleavg[mut[0]] = defaultdict(list)
        sampleavg[mut[0]][aft['time_binc']].append(aft['af'])
    stderr = defaultdict(list)
    for mut in sampleavg:
        for t in sorted(sampleavg[mut].keys()):
            stderr[mut].append(np.std(sampleavg[mut][t])/np.sqrt(len(sampleavg[mut][t])-1))

    if axs is None:
        savefig = True
        fig, axs = plt.subplots(1,2, figsize=(12,6))
    else:
        savefig = False

    mlist = ['A->G', 'C->T', 'G->A', 'T->C',
             'A->C', 'A->T', 'C->A', 'C->G',
             'G->T', 'G->C', 'T->G', 'T->A']
    for mut in mlist:
        aft = d.loc[mut]
        if mut in transitions:
            ax=axs[0]
            color = cmap[transitions.index(mut)]
            marker='o'
            ls='-o'
        else:
            ax=axs[1]
            if mut in transversions_pair:
                ls='--o'
                color = cmap[transversions_pair.index(mut)]
                marker='o'
            else:
                ls='-v'
                color = cmap[transversions_np.index(mut)]
                marker='v'

        times = np.array(aft.index) + 100*(np.random.random(size=len(aft))-0.5)
        aft = np.array(aft)
        if mu is None:
            label = mut[0] + u' \u2192 ' + mut[-1]
        else:
            label = None

        ax.errorbar(times, aft, np.array(stderr[mut]),
                    ls='none',
                    marker=marker,
                    markersize=10,
                    lw=3,
                    color=color)

        # Plot fit
        if mu is not None:
            xfit = np.array([-100,3000])
            yfit = xfit * mu.loc[mut]
            label = mut[0] + u' \u2192 ' + mut[-1]
            ax.plot(xfit, yfit,
                    ls,
                    lw=2.5,
                    color=color,
                    alpha=0.7,
                    label=label)

    for ax in axs:
        ax.legend(loc=2, ncol=2, numpoints=2, fontsize=16)
        ax.set_xlim([0,2700])
        ax.set_ylim(0)
        ax.set_xlabel('days since EDI', fontsize=16)
        ax.set_ylabel('fraction mutated', fontsize=16)
        ax.tick_params(axis='both', labelsize=18)
        ax.grid(True)
    axs[0].set_yticklabels(['0.00', '0.01', '0.02', '0.03', '0.04'])
    axs[1].set_yticklabels(['0.000', '0.002', '0.004', '0.006', '0.008'])

    plt.tight_layout()

    if savefig:
        plt.savefig('mutation_linear_increase.png')


def plot_mutation_rate_matrix(mu, dmulog10=None, savefig=False, ax=None):
    '''Plot mutation rate matrix'''
    from matplotlib import cm
    sns.set_style('dark')

    fig_width = 5
    fs = 16

    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_width, fig_width))
        ax.set_title(mu.name+'\n[$\log_{10}$ changes $\cdot$ day$^{-1}$]',
                     fontsize=fs)
    else:
        ax.set_title('$\log_{10}$ changes $\cdot$ day$^{-1}$', fontsize=fs)

    M = np.zeros((4, 4))
    for mut, rate in mu.iteritems():
        i_from = alphal.index(mut[0])
        i_to = alphal.index(mut[-1])
        M[i_from, i_to] = rate
    logM = np.log10(M)

    ax.imshow(logM, interpolation='nearest',
              cmap=cm.jet,
              vmin=-7.1, vmax=-4.5)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(3.5, -0.5)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xticklabels(alphal[:4])
    ax.set_yticklabels(alphal[:4])
    ax.xaxis.set_tick_params(labelsize=fs)
    ax.yaxis.set_tick_params(labelsize=fs)
    ax.set_ylabel('From', fontsize=fs)
    ax.set_xlabel('To', fontsize=fs)

    for mut, rate in mu.iteritems():
        i_from = alphal.index(mut[0])
        i_to = alphal.index(mut[-1])
        if -6.4 < np.log10(rate) < -5:
            color = 'black'
        else:
            color = 'white'

        txt = '{:1.1f}'.format(np.log10(rate))
        if dmulog10 is not None:
            txt = '$'+txt+' \pm '+'{:1.1f}'.format(dmulog10[mut])+'$'

        ax.text(i_to, i_from,
                txt,
                fontsize=fs-2,
                ha='center',
                va='center',
                color=color,
               )

    ax.grid(False)

    plt.tight_layout()

    if savefig:
        fig_filename = '/home/fabio/university/phd/thesis/tex/figures/mutation_rate_matrix_neutralclass'
        for ext in ['svg', 'pdf', 'png']:
            fig.savefig(fig_filename+'.'+ext)

    plt.ion()
    plt.show()


def plot_comparison(mu, muA, dmulog10=None, dmuAlog10=None, ax=None):
    '''Compare new estimate for mu with Abram et al 2010'''
    xmin = -7.3
    xmax = -4
    fs = 16

    if ax is None:
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

    ax.plot(np.linspace(xmin, xmax, 1000), np.linspace(xmin, xmax, 1000),
            color='grey',
            lw=1,
            alpha=0.7)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_xlabel(r'$\log_{10}(\mathrm{new\ estimate})$', fontsize=fs)
    ax.set_ylabel(r'$\log_{10}(\mathrm{Abram\ et\ al.\ 2010})$', fontsize=fs)
    ax.xaxis.set_tick_params(labelsize=fs)
    ax.yaxis.set_tick_params(labelsize=fs)
    ax.legend(loc=2, fontsize=fs)
    ax.grid(True)

    plt.tight_layout()

    plt.ion()
    plt.show()

    return ax


def plot_figure_1(data, mu, dmulog10, muA, dmuAlog10):
    '''Plot figure 1 of the paper'''
    print('Plot Figure 1')
    from figure_S1 import plot_mutation_rate as plot_mutation_rate_graph
    fig = plt.figure(figsize=(12, 11))
    ax1 = plt.subplot2grid((2,2), (0,0))
    ax2 = plt.subplot2grid((2,2), (0,1))
    ax3 = plt.subplot2grid((2,2), (1, 0), colspan=2)
    plot_mutation_increase(data, mu=mu, axs=[ax1, ax2])
    #plot_mutation_rate_matrix(mu, dmulog10=dmulog10, ax=axs[2])
    plot_mutation_rate_graph(mu, ax=ax3)
    #plot_comparison(mu, muA, dmulog10=dmulog10, dmuAlog10=dmuAlog10, ax=axs[3])

    # Add labels
    from util import add_panel_label
    add_panel_label(ax1, 'A', x_offset=-0.2)
    add_panel_label(ax2, 'B', x_offset=-0.2)
    add_panel_label(ax3, 'C', x_offset=-0.08)
    #add_panel_label(axs[3], 'D', x_offset=-0.2)

    for ext in ['svg', 'png', 'pdf']:
        fig.savefig('../figures/figure_1.'+ext)


def collect_data(patients, cov_min=100, refname='HXB2', subtype='any'):
    '''Collect data for the mutation rate estimate'''
    print('Collect data from patients')

    ref = HIVreference(refname=refname, load_alignment=True, subtype=subtype)

    data = []
    for pi, pcode in enumerate(patients):
        print(pcode)

        p = Patient.load(pcode)
        comap = (pd.DataFrame(p.map_to_external_reference('genomewide')[:, :2],
                              columns=[refname, 'patient'])
                   .set_index('patient', drop=True)
                   .loc[:, refname])

        aft = p.get_allele_frequency_trajectories('genomewide', cov_min=cov_min)
        times = p.dsi

        for pos, aft_pos in enumerate(aft.swapaxes(0, 2)):
            fead = p.pos_to_feature[pos]

            # Keep only sites within ONE protein
            if len(fead['protein_codon']) != 1:
                continue

            # Exclude codons with gaps
            pc = fead['protein_codon'][0][-1]
            cod_anc = ''.join(p.initial_sequence[pos - pc: pos - pc + 3])
            if '-' in cod_anc:
                continue

            for ia, aft_nuc in enumerate(aft_pos[:4]):
                # Keep only derived alleles
                if alpha[ia] == p.initial_sequence[pos]:
                    continue

                # Keep only no RNA structures
                if fead['RNA']:
                    continue

                # Keep only sites which are also in the reference
                if pos not in comap.index:
                    continue

                # Keep only high-entropy sites
                S_pos = ref.entropy[comap.loc[pos]]
                if S_pos < 0.1:
                    continue

                # Keep only synonymous alleles
                cod_new = cod_anc[:pc] + alpha[ia] + cod_anc[pc+1:]
                if translate(cod_anc) != translate(cod_new):
                    continue

                mut = p.initial_sequence[pos]+'->'+alpha[ia]

                for it, (t, af_nuc) in enumerate(izip(times, aft_nuc)):
                    # Keep only nonmasked times
                    if aft_nuc.mask[it]:
                        continue

                    datum = {'time': t,
                             'af': af_nuc,
                             'pos': pos,
                             'protein': fead['protein_codon'][0][0],
                             'pcode': pcode,
                             'mut': mut,
                             'subtype': subtype,
                             'refname': refname,
                            }
                    data.append(datum)

    data = pd.DataFrame(data)

    return data



# Script
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Mutation rate')
    parser.add_argument('--regenerate', action='store_true',
                        help="regenerate data from allele counts")
    args = parser.parse_args()

    # Intermediate data are saved to file for faster access later on
    fn = '../data/mutation_rate_data.pickle'
    patients = ['p1', 'p2', 'p3','p5', 'p6', 'p8', 'p9', 'p11']
    if not os.path.isfile(fn) or args.regenerate:
        data = collect_data(patients)
        try:
            data.to_pickle(fn)
            print('Data saved to file:', os.path.abspath(fn))
        except IOError:
            print('Could not save data to file:', os.path.abspath(fn))
    else:
        data = pd.read_pickle(fn)

    # Make time bins
    t_bins = np.array([0, 500, 1000, 1750, 3000], int)
    t_binc = 0.5 * (t_bins[:-1] + t_bins[1:])
    add_binned_column(data, t_bins, 'time')
    data['time_binc'] = t_binc[data['time_bin']]

    # Get mutation rate with bootstrap error bars
    mu,dmulog10 = get_mutation_matrix(data)

    # Compare to Abram et al 2010
    tmp = get_mu_Abram2010(with_std=True)
    muA = tmp['mu']
    dmuAlog10 = tmp['std'] / tmp['mu'] / np.log(10)

    # Save results to file (used in Figure 2)
    fn = '../data/mutation_rate.pickle'
    mu_out = pd.DataFrame({'mu': mu, 'muA': muA, 'dmulog10': dmulog10, 'dmuAlog10': dmuAlog10})
    mu_out.to_pickle(fn)

    # Plot Figure 1
    plot_figure_1(data=data, mu=mu, dmulog10=dmulog10, muA=muA, dmuAlog10=dmuAlog10)
