# vim: fdm=indent
'''
author:     Fabio Zanini
date:       15/06/15
content:    Make figure for the mutation rate.
'''
# Modules
import os
import sys
from itertools import izip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Bio.Seq import translate

from hivevo.patients import Patient
from hivevo.HIVreference import HIVreference
from hivevo.sequence import alpha, alphal



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


def add_binned_column(df, bins, to_bin):
    # FIXME: this works, but is a little cryptic
    df.loc[:, to_bin+'_bin'] = np.minimum(len(bins)-2,
                                          np.maximum(0,np.searchsorted(bins, df.loc[:,to_bin])-1))


def boot_strap_patients(df, eval_func, columns=None,  n_bootstrap=100):
    import pandas as pd

    if columns is None:
        columns = df.columns
    if 'pcode' not in columns:
        columns = list(columns)+['pcode']

    patients = df.loc[:,'pcode'].unique()
    tmp_df_grouped = df.loc[:,columns].groupby('pcode')
    npats = len(patients)
    replicates = []
    for i in xrange(n_bootstrap):
        if (i%20==0): print("Bootstrap",i)
        pats = patients[np.random.randint(0,npats, size=npats)]
        bs = []
        for pi,pat in enumerate(pats):
            bs.append(tmp_df_grouped.get_group(pat))
            bs[-1]['pcode']='BS'+str(pi+1)
        bs = pd.concat(bs)
        replicates.append(eval_func(bs))
    return replicates


def get_mutation_matrix(data):
    '''Calculate the mutation rate matrix'''
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


def plot_mutation_rate_matrix(mu, dmulog10=None, savefig=False):
    from matplotlib import cm
    sns.set_style('dark')

    fig_width = 5
    fs = 16

    fig, ax = plt.subplots(figsize=(fig_width, fig_width))

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
    ax.set_title(mu.name+'\n[$\log_{10}$ changes $\cdot$ day$^{-1}$]',
                 fontsize=fs)

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

    plt.tight_layout()
    
    if savefig:
        fig_filename = '/home/fabio/university/phd/thesis/tex/figures/mutation_rate_matrix_neutralclass'
        for ext in ['svg', 'pdf', 'png']:
            fig.savefig(fig_filename+'.'+ext)

    plt.ion()
    plt.show()


def plot_comparison(mu, muA, dmulog10=None, dmuAlog10=None):
    '''Compare new estimate for mu with Abram et al 2010'''
    xmin = -7.3
    xmax = -4
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

    label = 'Pearson r = {0:3.0%},\nSpearman r = {1:3.0%}'.format(R, rho)

    ax.errorbar(x, y,
                xerr=dx, yerr=dy,
                ls='o',
                ms=10,
                marker='o',
                label=label)

    ax.plot(np.linspace(xmin, xmax, 1000), np.linspace(xmin, xmax, 1000),
            color='grey',
            lw=1,
            alpha=0.7)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_xlabel('log10 (new estimate)', fontsize=fs)
    ax.set_ylabel('log10 (Abram et al. 2010)', fontsize=fs)
    ax.xaxis.set_tick_params(labelsize=fs)
    ax.yaxis.set_tick_params(labelsize=fs)
    ax.legend(loc=2, fontsize=fs)
    ax.grid(True)

    plt.tight_layout()

    plt.ion()
    plt.show()


def collect_data(patients, cov_min=100):
    '''Collect data for the mutation rate estimate'''
    ref = HIVreference(load_alignment=False)

    data = []
    for pi, pcode in enumerate(patients):
        print pcode

        p = Patient.load(pcode)
        comap = (pd.DataFrame(p.map_to_external_reference('genomewide')[:, :2],
                              columns=['HXB2', 'patient'])
                   .set_index('patient', drop=True)
                   .loc[:, 'HXB2'])

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

                # Keep only high-entropy sites
                if pos not in comap.index:
                    continue
                S_pos = ref.entropy[comap.loc[pos]]
                if S_pos < 0.01:
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
                            }
                    data.append(datum)

    data = pd.DataFrame(data)

    return data



# Script
if __name__ == '__main__':

    patients = ['p1', 'p2', 'p3','p5', 'p6', 'p8', 'p9', 'p11']
    cov_min = 100

    data = collect_data(patients, cov_min=cov_min)

    t_bins = np.array([0, 500, 1000, 1500, 2000, 3000], int)
    t_binc = 0.5 * (t_bins[:-1] + t_bins[1:])
    add_binned_column(data, t_bins, 'time')
    data['time_binc'] = t_binc[data['time_bin']]

    mu = get_mutation_matrix(data)

    if True:
        dmulog10 = mu.copy()
        muBS = boot_strap_patients(data, get_mutation_matrix, n_bootstrap=100)
        for key, _ in dmulog10.iteritems():
            dmulog10[key] = np.std([np.log10(tmp[key]) for tmp in muBS])
    else:
        dmulog10 = None

    plot_mutation_rate_matrix(mu, dmulog10=dmulog10)

    # Compare to Abram et al 2010
    tmp = get_mu_Abram2010(with_std=True)
    muA = tmp['mu']
    dmuAlog10 = tmp['std'] / tmp['mu'] / np.log(10)
    plot_mutation_rate_matrix(muA, dmulog10=dmuAlog10)

    plot_comparison(mu, muA, dmulog10=dmulog10, dmuAlog10=dmuAlog10)


    # Save to file
    fn = '../data/mutation_rate.pickle'
    a = pd.DataFrame({'mu': mu, 'muA': muA, 'dmulog10': dmulog10, 'dmuAlog10': dmuAlog10})
