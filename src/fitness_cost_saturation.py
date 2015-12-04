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
def load_mutation_rates():
    fn = 'data/mutation_rate.pickle'
    return pd.read_pickle(fn)


def fit_fitness_cost(data, mu='muAbram', independent_slope=True):
    '''Get fitness costs from the data'''
    # Group by time (binned) and average, then fit nonlinear least squares
    d = (data
         .loc[:, ['time_binc', 'S_binc', mu, 'af']]
         .groupby([mu, 'S_binc', 'time_binc'], as_index=False)
         .mean()
         .groupby('S_binc'))

    from scipy.optimize import curve_fit
    if not independent_slope:
        # x[0] is time, x[1] the mutation rate
        fun = lambda x, s: x[1] / s * (1 - np.exp(-s * x[0]))
        s = {}
        for iS, (S, datum) in enumerate(d):
            x = np.array(datum[['time_binc', mu]]).T
            y = np.array(datum['af'])

            ind = np.array(datum[mu] > 1e-5)
            x = x[:, ind]
            y = y[ind]

            s0 = 1e-2 / 10**(iS / 2.0)
            fit = curve_fit(fun, x, y, [s0])
            s[S] = fit[0][0]
        s = pd.Series(s)
    else:
        # x[0] is time, x[1] the mutation rate
        fun = lambda x, s1, s2: x[1] / s1 * (1 - np.exp(-s2 * x[0]))
        s = {}
        for iS, (S, datum) in enumerate(d):
            x = np.array(datum[['time_binc', mu]]).T
            y = np.array(datum['af'])

            ind = np.array(datum[mu] > 1e-5)
            x = x[:, ind]
            y = y[ind]

            s0 = 1e-2 / 10**(iS / 2.0)
            fit = curve_fit(fun, x, y, [s0, s0])
            s[S] = {'s1': fit[0][0], 's2': fit[0][1]}
        s = pd.DataFrame(s).T

    s.index.name = 'Subtype entropy'
    s.name = 'Fitness cost'

    return s


def fit_fitness_cost_mu(data):
    '''Fit the fitness costs and the mutation rate together'''
    from scipy.optimize import curve_fit

    # Poor man's version: we fit mu as a linear increase in the most variable
    # entropy class, then we fix it for all classes. A better version would be
    # an actual multidimensional minimization, but good enough for now
    d = (data
         .loc[data['S_binc'] == data['S_binc'].max()]
         .loc[:, ['time_binc', 'mut', 'af']]
         .groupby(['mut', 'time_binc'])
         .mean()
         .loc[:, 'af']
         .unstack('time_binc'))

    mu = {}
    for mut, datum in d.iterrows():
        x = np.array(datum.index)
        y = np.array(datum)
        m = np.dot(x, y) / np.dot(x, x)
        mu[mut] = m
    mu = pd.Series(mu, name='mutation rate, from variable sites')

    # Group by time (binned) and average, then fit nonlinear least squares
    d = (data
         .loc[:, ['time_binc', 'S_binc', 'af', 'mut']]
         .groupby(['mut', 'S_binc', 'time_binc'], as_index=False)
         .mean()
         .groupby('S_binc'))

    # x[0] is time, x[1] the mutation rate
    fun = lambda x, s: x[1] / s * (1 - np.exp(-s * x[0]))
    s = {}
    for iS, (S, datum) in enumerate(d):
        x = np.vstack([np.array(datum['time_binc']), np.array(mu[datum['mut']])])
        y = np.array(datum['af'])

        s0 = 1e-2 / 10**(iS / 2.0)
        fit = curve_fit(fun, x, y, [s0])
        s[S] = fit[0][0]
    s = pd.Series(s)

    return s, mu


    # x[0] is time, x[1] the mutation rate
    fun = lambda x, s: x[1] / s * (1 - np.exp(-s * x[0]))
    s = {}
    for iS, (S, datum) in enumerate(d):
        x = np.array(datum[['time_binc', mu]]).T
        y = np.array(datum['af'])

        ind = np.array(datum[mu] > 1e-5)
        x = x[:, ind]
        y = y[ind]

        s0 = 1e-2 / 10**(iS / 2.0)
        fit = curve_fit(fun, x, y, [s0])
        s[S] = fit[0][0]
    s = pd.Series(s)


def plot_fits_4x4(data, s, mu):
    '''Plot the fit curves for all mutations'''
    from matplotlib import cm
    sns.set_style('darkgrid')

    fig_width = 3
    fs = 16
    fig, axs = plt.subplots(4, 4,
                            figsize=(4 * fig_width, 4 * fig_width))

    for ia1, a1 in enumerate(alphal[:4]):
        for ia2, a2 in enumerate(alphal[:4]):
            mut = a1+'->'+a2
            ax = axs[ia1, ia2]
            if ia1 == ia2:
                ax.xaxis.set_ticks([])
                ax.yaxis.set_ticks([])
                ax.text(0.5, 0.5, a1, fontsize=fs+6, ha='center', va='center')
                continue

            # Trajectories
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
            ax.set_ylim(-0.005, 0.08)
            ax.set_xlim(0, 3000)
            if (ia1, ia2) != (3, 0):
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])

            # Fits
            if isinstance(s, pd.Series):
                fun = lambda t, s: mu[mut] / s * (1 - np.exp(-s * t))
                for iS in xrange(d.ngroups):
                    xfit = np.linspace(0, x.max())
                    yfit = fun(xfit, s.iloc[iS])
                    ax.plot(xfit, yfit, lw=2, color=colors[iS], alpha=0.5)

            else:
                fun = lambda t, s1, s2: mu[mut] / s1 * (1 - np.exp(-s2 * t))
                for iS in xrange(d.ngroups):
                    xfit = np.linspace(0, x.max())
                    yfit = fun(xfit, s.iloc[iS]['s1'], s.iloc[iS]['s2'])
                    ax.plot(xfit, yfit, lw=2, color=colors[iS], alpha=0.5)

            # Pure mutation rate accumulation
            xfit = np.linspace(0, 3000)
            yfit = mu[mut] * xfit
            ax.plot(xfit, yfit, lw=2, color='grey', alpha=0.7)


    ax = axs[-1, 0]
    ax.set_xlabel('Time [days from infection]', fontsize=fs)
    ax.set_ylabel('Average allele frequency', fontsize=fs)
    ax.set_xticks(np.linspace(0, 0.08, 5))
    ax.set_xticks([0, 1000, 2000, 3000])
    ax.xaxis.set_tick_params(labelsize=fs)
    ax.yaxis.set_tick_params(labelsize=fs)

    fig.text(0.53, 0.03, 'To', fontsize=20, ha='center', va='center')
    fig.text(0.03, 0.53, 'From', fontsize=20, ha='center', va='center', rotation=90)

    plt.tight_layout()
    plt.ion()
    plt.show()


def plot_fitness_cost(data, s, mu, ds=None, mut='A->G', savefig=False):
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
    if isinstance(s, pd.Series):
        fun = lambda t, s: mu[mut] / s * (1 - np.exp(-s * t))
        for iS in xrange(d.ngroups):
            xfit = np.linspace(0, x.max())
            yfit = fun(xfit, s.iloc[iS])
            ax.plot(xfit, yfit, lw=2, color=colors[iS], alpha=0.5)
    else:
        fun = lambda t, s1, s2: mu[mut] / s1 * (1 - np.exp(-s2 * t))
        for iS in xrange(d.ngroups):
            xfit = np.linspace(0, x.max())
            yfit = fun(xfit, s.iloc[iS]['s1'], s.iloc[iS]['s2'])
            ax.plot(xfit, yfit, lw=2, color=colors[iS], alpha=0.5)

    # Costs
    if not isinstance(s, pd.Series):
        s = s['s2']

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


def collect_data(patients, cov_min=100, no_sweeps=False, refname='HXB2'):
    '''Collect data for the fitness cost estimate'''
    ref = HIVreference(refname=refname, subtype='any', load_alignment=True)
    mus = load_mutation_rates()
    mu = mus.mu
    muA = mus.muA

    data = []
    for pi, pcode in enumerate(patients):
        print pcode

        p = Patient.load(pcode)
        comap = (pd.DataFrame(p.map_to_external_reference('genomewide', refname=refname)[:, :2],
                              columns=[refname, 'patient'])
                   .set_index('patient', drop=True)
                   .loc[:, refname])

        aft = p.get_allele_frequency_trajectories('genomewide', cov_min=cov_min)
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

            # Keep only nonmasked times
            if aft_pos[:4].mask.any(axis=0).all():
                continue
            else:
                ind = ~aft_pos[:4].mask.any(axis=0)
                times = p.dsi[ind]
                aft_pos = aft_pos[:, ind]

            # Get site entropy
            if pos not in comap.index:
                continue
            pos_ref = comap.loc[pos]
            S_pos = ref.entropy[pos_ref]

            # Keep only sites where the ancestral allele and group M agree
            if ref.consensus_indices[pos_ref] != aft_pos[:, 0].argmax():
                continue

            for ia, aft_nuc in enumerate(aft_pos[:4]):
                # Keep only derived alleles
                if alpha[ia] == p.initial_sequence[pos]:
                    continue

                # Filter out sweeps if so specified
                if no_sweeps and (aft_nuc > 0.5).any():
                    continue

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


def cross_check_mu(data):
    '''Estimate the mutation rate from the high-entropy class'''
    from mutation_rate import get_mutation_matrix, plot_comparison

    # Get other estimate and Abram 2010
    tmp = load_mutation_rates()
    muO = tmp['mu']
    dmuOlog10 = tmp['dmulog10']
    muA = tmp['muA']
    dmuAlog10 = tmp['dmuAlog10']

    # Only high entropy
    data_mu = data.loc[data['S_binc'] == S_binc[-1]].copy()
    mu = get_mutation_matrix(data_mu)
    ax = plot_comparison(mu, muO, dmulog10=None, dmuAlog10=dmuOlog10)
    ax.set_title('Only high-entropy')
    ax.set_ylabel('Direct estimate of ours')

    # Difference
    ax = plot_comparison(mu - muO, muO, dmulog10=None, dmuAlog10=dmuOlog10)
    ax.set_title('Only high-entropy')
    ax.set_ylabel('Direct estimate of ours')
    ax.set_xlabel('Additional slope')

    return

    # High entropy and synonymous
    data_mu = data.loc[((data['S_binc'] == S_binc[-1]) &
                        (data['syn'] == True))].copy()
    mu = get_mutation_matrix(data_mu)
    ax = plot_comparison(mu, muO, dmulog10=None, dmuAlog10=dmuOlog10)
    ax.set_title('Only high-entropy AND synonymous')
    ax.set_ylabel('Direct estimate of ours')

    # See what happens for high-S, nonsynonymous sites: are they sweeps
    for iS in [-1, -2, -3]:
        dataS = data.loc[((data['S_binc'] == S_binc[iS]) &
                            (data['syn'] == False))].copy()
        g = (dataS.loc[:, ['af', 'time', 'pcode', 'pos_ref', 'mut']]
             .groupby(['pcode', 'pos_ref', 'mut']))

        n_fix = 0
        fig, ax = plt.subplots()
        for (pcode, pos, mut), datum in g:
            x = np.array(datum['time'])
            y = np.array(datum['af'])
            if (y > 0).any():
                ax.plot(x, y, lw=2)

            if (y > 0.8).any():
                n_fix += 1


        ax.set_xlabel('Time [days since EDI]')
        ax.set_ylabel('$\\nu$')
        if iS == -1:
            label = 'Highest'
        elif iS == -2:
            label = '2nd-highest'
        elif iS == -3:
            label = '3rd-highest'
        else:
            label = str(-iS)+'th highest'
        label += ' entropy, N fix: '+str(n_fix)+' out of '+str(g.ngroups)
        print label
        ax.set_title(label)

    plt.ion()
    plt.show()




# Script
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fitness cost')
    parser.add_argument('--regenerate', action='store_true',
                        help="regenerate data")
    parser.add_argument('--no-sweeps', action='store_true',
                        help='Exclude sweeps from the data collection')
    args = parser.parse_args()

    fn = 'data/fitness_cost_data.pickle'
    if args.no_sweeps:
        fn = fn.split('.')[0]+'_nosweep.pickle'
    if not os.path.isfile(fn) or args.regenerate:
        patients = ['p1', 'p2', 'p3','p5', 'p6', 'p8', 'p9', 'p11']
        cov_min = 100
        data = collect_data(patients, cov_min=cov_min, no_sweeps=args.no_sweeps)
        data.to_pickle(fn)
    else:
        data = pd.read_pickle(fn)
    sys.exit()

    # Make time and entropy bins
    t_bins = np.array([0, 500, 1000, 1500, 2000, 3000], int)
    t_binc = 0.5 * (t_bins[:-1] + t_bins[1:])
    add_binned_column(data, t_bins, 'time')
    data['time_binc'] = t_binc[data['time_bin']]

    S_bins = np.percentile(data['S'], np.linspace(0, 100, 8))
    S_binc = 0.5 * (S_bins[:-1] + S_bins[1:])
    add_binned_column(data, S_bins, 'S')
    data['S_binc'] = S_binc[data['S_bin']]

    # As a cross check, estimate the mutation rate from the high-entropy class
    cross_check_mu(data)

    sys.exit()

    # Set mutation rates
    mu = data.loc[:, ['mut', 'mu']].groupby('mut').mean()['mu']
    muA = data.loc[:, ['mut', 'muAbram']].groupby('mut').mean()['muAbram']

    s = fit_fitness_cost(dataS)

    s, mu = fit_fitness_cost_mu(dataS)
    sys.exit()

    if False:
        ds = s.copy()
        sBS = boot_strap_patients(data, fit_fitness_cost, n_bootstrap=100)
        for key, _ in s.iteritems():
            ds[key] = np.std([tmp[key] for tmp in sBS])
    else:
        ds = None

    plot_fitness_cost(data, s, mu, ds=ds)
