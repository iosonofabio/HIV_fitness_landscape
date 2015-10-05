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
def load_mutation_rates():
    fn = '../data/mutation_rate.pickle'
    return pd.read_pickle(fn)


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


def get_fitness_cost(data, mu='muAbram'):
    '''Get fitness costs from the data'''
    # Group for the fit
    d = (data
         .loc[:, ['time_binc', 'S_binc', mu, 'af']]
         .groupby([mu, 'S_binc', 'time_binc'], as_index=False)
         .mean()
         .groupby('S_binc'))

    # Fit
    from scipy.optimize import curve_fit
    # x[0] is time, x[1] the mutation rate
    fun = lambda x, s: x[1] / s * (1 - np.exp(-s * x[0]))
    s = {}
    for iS, (S, datum) in enumerate(d):
        x = np.array(datum[['time_binc', mu]]).T
        y = np.array(datum['af'])
        s0 = 1e-2 / 10**(iS / 2.0)
        fit = curve_fit(fun, x, y, [s0])
        s[S] = fit[0][0]

    s = pd.Series(s)
    s.index.name = 'Subtype entropy'
    s.name = 'Fitness cost'

    return s


def plot_fitness_cost(data, s, mu, ds=None, mut='G->A', savefig=False):
    '''Plot the fitness costs'''
    from matplotlib import cm
    sns.set_style('darkgrid')

    fig_width = 5
    fs = 16

    fig, axs = plt.subplots(1, 2,
                            figsize=(2 * fig_width, fig_width))

    # Trajectories: group for the plot, only highest mutation rate
    ax = axs[0]
    d = (data
         .groupby('mut')
         .get_group(mut)
         .loc[:, ['time_binc', 'S_binc', 'af']]
         .groupby(['S_binc', 'time_binc'], as_index=False)
         .mean()
         .groupby('S_binc'))
    colors = [cm.jet(1.0 * iS / d.ngroups) for iS in xrange(d.ngroups)]
    for iS, (S, datum) in enumerate(d):
        x = np.array(datum['time_binc'])
        y = np.array(datum['af'])
        ax.scatter(x, y,
                   s=70,
                   color=colors[iS],
                   label='{:1.1G}'.format(S),
                  )
    ax.set_xlabel('Time [days from infection]', fontsize=fs)
    ax.set_ylabel('Average allele frequency', fontsize=fs)
    ax.set_ylim(-0.005, 0.08)
    ax.set_xticks(np.linspace(0, 0.08, 5))
    ax.set_xticks([0, 1000, 2000, 3000])
    ax.xaxis.set_tick_params(labelsize=fs)
    ax.yaxis.set_tick_params(labelsize=fs)

    text_label = 'Mutation: '+mut[0]+' > '+mut[-1]
    ax.text(0.05, 0.95,
            text_label,
            transform=ax.transAxes,
            ha='left', va='top')

    # Fits
    fun = lambda t, s: mu[mut] / s * (1 - np.exp(-s * t))
    for iS in xrange(d.ngroups):
        xfit = np.linspace(0, x.max())
        yfit = fun(xfit, s.iloc[iS])
        ax.plot(xfit, yfit, lw=2, color=colors[iS], alpha=0.5)


    # Costs
    ax = axs[1]
    x = np.array(s.index)
    y = -np.array(s)
    if ds is not None:
        dy = np.array(ds)
    if x[0] == 0:
        x = x[1:]
        y = y[1:]
        if ds is not None:
            dy = dy[1:]

        add_at_zero = True
    else:
        add_at_zero = False

    if ds is not None:
        ax.errorbar(x, y,
                    yerr=dy,
                    lw=2,
                    color='k',
                   )
    else:
        ax.plot(x, y,
                lw=2,
                color='k',
               )
    if add_at_zero:
        ax.plot([1e-3, s.index[1]],
                [-s.iloc[0], -s.iloc[1]],
                lw=2,
                ls='--',
                color='k',
               )
        ax.annotate('Full conservation',
                    xy=(1.1e-3, -s.iloc[0]),
                    xytext=(5e-2, -s.iloc[0]),
                    arrowprops={'facecolor': 'black',
                                'width': 1.5,
                                'frac': 0.4,
                                'shrink': 0.1,
                               },
                    ha='center',
                    va='center',
                    fontsize=fs,
                   )

    ax.set_xlabel('Variability in subtype B [bits]', fontsize=fs)
    ax.set_ylabel('Fitness effect', fontsize=fs)
    ax.set_xlim(1e-3, 1)
    ax.set_ylim(-0.08, 0.001)
    ax.set_xscale('log')
    ax.xaxis.set_tick_params(labelsize=fs)
    ax.yaxis.set_tick_params(labelsize=fs)

    plt.tight_layout()


    if savefig:
        fig_filename = '/home/fabio/university/phd/thesis/tex/figures/fitness_cost'
        for ext in ['svg', 'pdf', 'png']:
            fig.savefig(fig_filename+'.'+ext)

    plt.ion()
    plt.show()


def collect_data(patients, cov_min=100):
    '''Collect data for the fitness cost estimate'''
    ref = HIVreference(load_alignment=False)
    mus = load_mutation_rates()
    mu = mus.mu
    muA = mus.muA

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
            # Note: we could drop this, but then we cannot quite classify syn/nonsyn
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

                # Get site entropy
                if pos not in comap.index:
                    continue
                pos_ref = comap.loc[pos]
                S_pos = ref.entropy[pos_ref]

                # Annotate with syn/nonsyn alleles
                cod_new = cod_anc[:pc] + alpha[ia] + cod_anc[pc+1:]
                if translate(cod_anc) != translate(cod_new):
                    syn = False
                else:
                    syn = True

                mut = p.initial_sequence[pos]+'->'+alpha[ia]
                mu_pos = mu[mut]
                muA_pos = muA[mut]

                for it, (t, af_nuc) in enumerate(izip(times, aft_nuc)):
                    # Keep only nonmasked times
                    if aft_nuc.mask[it]:
                        continue

                    datum = {'time': t,
                             'af': af_nuc,
                             'pos': pos,
                             'pos_ref': pos_ref,
                             'protein': fead['protein_codon'][0][0],
                             'pcode': pcode,
                             'mut': mut,
                             'mu': mu_pos,
                             'muAbram': muA_pos,
                             'S': S_pos,
                             'syn': syn,
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

    S_bins = np.percentile(data['S'], np.linspace(0, 100, 8))
    S_binc = 0.5 * (S_bins[:-1] + S_bins[1:])
    add_binned_column(data, S_bins, 'S')
    data['S_binc'] = S_binc[data['S_bin']]

    s = get_fitness_cost(data)
    sBS = boot_strap_patients(data, get_fitness_cost, n_bootstrap=100)
    ds = s.copy()
    for key, _ in s.iteritems():
        ds[key] = np.std([tmp[key] for tmp in sBS])

    mu = data.loc[:, ['mut', 'mu']].groupby('mut').mean()['mu']
    plot_fitness_cost(data, s, mu, ds=ds)
