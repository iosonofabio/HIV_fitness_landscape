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


def interpolate_slope(mu, muNS, nu):
    # linear
    #return mu + nu * (muNS - mu)

    # log
    #return muNS * (mu / muNS)**(1.0 - nu)

    # Theta
    theta = 0.2
    return muNS + (nu < theta) * (mu - muNS)



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


def fit_mu_highentropy(data):
    '''An estimate of the high-entropy initial slope is necessary since we exclude sweeps'''
    from mutation_rate import get_mutation_matrix, plot_comparison

    data_mu = data.loc[data['S_binc'] ==  data['S_binc'].max()].copy()
    mu = get_mutation_matrix(data_mu)
    mu.name = 'mutation rate, from high-entropy sites'
    return mu


def compare_mu(mu, plot=False):
    '''Compare new estimate for mu with the proper one'''
    from mutation_rate import get_mutation_matrix, plot_comparison

    # Get other estimate and Abram 2010
    tmp = load_mutation_rates()
    muO = tmp['mu']
    dmuOlog10 = tmp['dmulog10']
    muA = tmp['muA']
    dmuAlog10 = tmp['dmuAlog10']

    # Fit a proportionality factor (weighted geometric mean of ratios)
    ratios = np.log10(mu / muO)
    weights = 1.0 / dmuOlog10
    alpha = 10**(np.dot(ratios, weights) / np.sum(weights))

    if plot:
        ax = plot_comparison(mu, muO, dmulog10=None, dmuAlog10=dmuOlog10)
        ax.set_title('Only high-entropy')
        ax.set_ylabel('Direct estimate of ours')
        x = np.linspace(-7.5, -4, 100)
        y = x - np.log10(alpha)
        ax.plot(x, y, lw=2, color='red',
                label='$\\alpha = '+'{:1.2f}'.format(alpha)+'$')
        ax.legend(loc='upper left', fontsize=18)

    return alpha


def prepare_data_for_fit(data, plot=False):
    '''Prepare the data for fit by averaging single allele frequencies'''
    def plot_bins(counter):
        fig, ax = plt.subplots()
        h, bins = np.histogram(counter, bins=10)
        for i in xrange(len(h)):
            ax.bar(bins[i], h[i], bins[i+1] - bins[i])
        ax.set_xlabel('# points in average frequency')
        cmin = counter.min()
        cmax = counter.max()
        ax.annotate('min = '+str(int(cmin)),
                    (cmin, h[0] * 1.02),
                    (cmin * 1.3, h[0] * 1.2),
                    arrowprops={'facecolor': 'black',
                                'width': 1.5,
                                'headlength': 10,
                                'shrink': 0.1,
                               },
                    ha='center',
                    va='center',
                    fontsize=18,
                   )
        plt.ion()
        plt.show()

    # Set std devs as a max of binomial sampling and sequencing error
    data['af_std'] = np.maximum(np.sqrt(data['af'] * (1 - data['af']) / data['n_templates']),
                                1e-3)
    data['af_weight'] = 1
    data['af_weighted'] = data['af'] * data['af_weight']
    data['counter'] = 1.0

    cols = ['mut', 'S_binc', 'time_binc', 'af', 'af_weight', 'af_weighted', 'counter']

    # Assuming different samples have unrelated errors, the variance is averaged
    data['af_std2_weighted'] = data['af_std']**2 * data['af_weight']
    cols.append('af_std2_weighted')

    # Group by time (binned) and average
    def average_data(data):
        d = (data
             .loc[:, cols]
             .groupby(['mut', 'S_binc', 'time_binc'], as_index=True)
             .sum())

        d['af_simple'] = d['af'] / d['counter']
        d['af'] = d['af_weighted'] / d['af_weight']
        return d
    
    d = average_data(data)
    if plot:
        plot_bins(d['counter'])

    # Errors on the mean (potentially much smaller than 1e-3)
    d['af_std'] = np.sqrt(d['af_std2_weighted'] / d['af_weight'] / (d['counter'] - 1))

    del d['af_weight']
    del d['af_weighted']
    del d['af_std2_weighted']
    del d['counter']
    return d


def fit_fitness_cost_interpmu(data_to_fit, mu,
                              muNS,
                              nu_sweep_norm):
    '''Fit the fitness costs with an interpolated initial slope
    
    The rationale for this is that the simple fit

        <nu> (t) = mu / s * (1 - exp[-s t])

    does not work because of selective sweeps. After filtering out
    fixing sites this estimate of mu becomes somewhat proportional, but lower
    than the actual mutation rate (it is expected, we are removing some
    successful hitchhikers): the ratio is around 0.4.

    Because the exclusion of sweeps affects mostly high-entropy classes, we
    interpolate between the actual mutation rate 'mu' and the lower-proportional
    'muNS' according to entropy class (low-S -> mu, high-S -> muNS). The exact
    nature of the interpolation is not known, but it does not make a huge difference
    because alpha is relatively close to 1.
    '''
    from scipy.optimize import curve_fit

    data_to_fit = data_to_fit.copy()
    data_to_fit['mut'] = data_to_fit.index.get_level_values('mut')
    data_to_fit['time_binc'] = data_to_fit.index.get_level_values('time_binc')
    d = data_to_fit.groupby(level='S_binc')

    # x[0] is time, x[1] the mutation rate
    #fig, ax = plt.subplots()
    s = []
    for iS, (S, datum) in enumerate(d):
        mut = datum['mut']
        nu_sweep_tmp = nu_sweep_norm.iloc[:, iS][mut]

        # Decrease initial slope by linear interpolation of the bias due to
        # exclusion of sweeps, anchored at the high-entropy class
        mu_tmp = interpolate_slope(mu[mut], muNS[mut], nu_sweep_tmp)

        x = np.vstack([np.array(datum['time_binc']),
                       np.array(mu_tmp),
                       ])
        y = np.array(datum['af'])
        dy = np.array(datum['af_std'])

        s0 = 1e-2 / 10**(iS / 2.0)
        fun = lambda x, s: x[1] / s * (1 - np.exp(-s * x[0]))
        fit = curve_fit(fun, x, y, p0=[s0], sigma=dy, absolute_sigma=False)
        s.append({'S': S,
                  's': fit[0][0],
                  'ds': np.sqrt(fit[1][0, 0]),
                 })

        if False:
            from matplotlib import cm
            mut = 'C->T'
            dd = datum.loc[datum['mut'] == mut]
            x = np.array(dd['time_binc'])
            y = np.array(dd['af'])
            dy = np.array(dd['af_std'])
            y2 = np.array(dd['af'])
            ax.errorbar(x, y,
                        yerr=dy, lw=2,
                        color=cm.jet(1.0 * iS / 6))
            ax.plot(x, y2, lw=2, ls='--',
                    color=cm.jet(1.0 * iS / 6))

            #xfit = np.linspace(0, 3000)
            #xx = np.vstack([xfit, np.repeat(mu_tmp.loc[mut].iloc[0], len(xfit))])
            #yfit = fun(xx, fit[0][0])
            #ax.plot(xfit, yfit, lw=2,
            #        color=cm.jet(1.0 * iS / 6))

            #import ipdb; ipdb.set_trace()

    s = pd.DataFrame(s).set_index('S', inplace=False, drop=True)
    s.index.name = 'entropy'
    s.name = 'fitness cost'

    return s


def fit_fitness_cost_mu(data, alpha=None):
    '''Fit the fitness costs and the mutation rate together
    
    The rationale for this is that the simple fit
        <nu> (t) = mu / s * (1 - exp[-s t])
    does not work because of selective sweeps. After filtering out
    fixing sites this estimate of mu becomes somewhat proportional, but lower
    than the actual mutation rate (it is expected, we are removing some
    successful hitchhikers).
    
    Of course, only after a sensible estimate of the initial linear increase does
    the saturation fit make any sense, so we fit both here (the space of
    parameters is the CARTESIAN product, not the TENSOR product, so we have 12
    more parameters only).
    '''
    from scipy.optimize import curve_fit

    # Poor man's version: we fit mu as a linear increase in the most variable
    # entropy class, then we fix it for all classes. A better version would be
    # an actual multidimensional minimization, but that would also require
    # legitimate error bars on rare frequencies, which we are not providing really
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
    s = {}
    for iS, (S, datum) in enumerate(d):
        x = np.vstack([np.array(datum['time_binc']), np.array(mu[datum['mut']])])
        y = np.array(datum['af'])
        s0 = 1e-2 / 10**(iS / 2.0)
        fun = lambda x, s: 1.0 / al * x[1] / s * (1 - np.exp(-s * x[0]))
        fit = curve_fit(fun, x, y, [s0])
        s[S] = fit[0][0]
    s = pd.Series(s)
    s.index.name = 'entropy'
    s.name = 'fitness cost'

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


def fit_all_transitions(data):
    from scipy.optimize import curve_fit
    res = []
    muts = ['A->G', 'G->A', 'C->T', 'T->C']
    for mut in muts:
        d = (data.groupby('mut')
             .get_group(mut)
             .loc[:, ['time_binc', 'S_binc', 'af']]
             .groupby(['S_binc', 'time_binc'], as_index=False)
             .mean()
             .groupby('S_binc'))

        #from matplotlib import cm
        #fig, ax = plt.subplots()
        #ax.set_title(mut)
        #ax.set_xlabel('Time since EDI [days]')
        #ax.set_ylim(0, 0.008)
        #xfit = np.linspace(0, 3000, 200)

        for iS, (S, datum) in enumerate(d):
            x = np.array(datum['time_binc'])
            y = np.array(datum['af'])

            fun = lambda x, s, mu: mu / s * (1 - np.exp(-s * x))
            s0 = 1e-3 / 10**(iS / 4.0)
            mu0 = 3e-6
            fit = curve_fit(fun, x, y, [s0, mu0])
            stmp = fit[0][0]
            mutmp = fit[0][1]

            #yfit = fun(xfit, stmp, mutmp)
            #ax.scatter(x, y, lw=2, color=cm.jet(1.0 * iS / 7))
            #ax.plot(xfit, yfit, lw=2, color=cm.jet(1.0 * iS / 7))

            res.append({'mut': mut,
                        'S': S,
                        'mu': mutmp,
                        's': stmp})

        #plt.ion()
        #plt.show()
        #import ipdb; ipdb.set_trace()

    res = pd.DataFrame(res).set_index(['S', 'mut'], inplace=False, drop=True)
    return res['s'].unstack(), res['mu'].unstack()


def plot_fitness_cost(data_to_fit,
                      s, mu, ds=None, mut='A->G',
                      alpha=None,
                      nu_sweep_norm=None,
                      muNS=None,
                      savefig=False):
    '''Plot the fitness costs'''
    from matplotlib import cm
    sns.set_style('darkgrid')

    fig_width = 5
    fs = 16

    fig, axs = plt.subplots(1, 2,
                            figsize=(2 * fig_width, fig_width))

    # Trajectories: group for the plot, only one mutation rate
    ax = axs[0]
    data_to_fit = data_to_fit.copy()
    data_to_fit['time_binc'] = data_to_fit.index.get_level_values('time_binc')
    d = (data_to_fit.groupby(level='mut')
         .get_group(mut)
         .groupby(level='S_binc'))

    colors = [cm.jet(1.0 * iS / d.ngroups) for iS in xrange(d.ngroups)]
    for iS, (S, datum) in enumerate(d):
        x = np.array(datum['time_binc'])
        y = np.array(datum['af'])
        dy = np.array(datum['af_std'])
        ax.scatter(x, y,
                   s=70,
                   color=colors[iS],
                   label='{:1.1G}'.format(S),
                  )
        ax.errorbar(x, y, yerr=dy,
                    fmt='none',
                    ecolor=colors[iS],
                  )
    ax.set_xlabel('Time [days from infection]', fontsize=fs)
    ax.set_ylabel('Average allele frequency', fontsize=fs)
    ax.set_ylim(-0.003, 0.01)
    ax.set_xticks(np.linspace(0, 0.08, 5))
    ax.set_xticks([0, 1000, 2000, 3000])
    ax.xaxis.set_tick_params(labelsize=fs)
    ax.yaxis.set_tick_params(labelsize=fs)

    text_label = 'Mutation: '+mut[0]+' -> '+mut[-1]
    ax.text(0.05, 0.95,
            text_label,
            transform=ax.transAxes,
            ha='left', va='top')

    # Fits
    for iS in xrange(d.ngroups):
        fun = lambda t, mu, s: mu / s * (1 - np.exp(-s * t))

        mu_tmp = mu[mut]#.iloc[iS]
        if nu_sweep_norm is not None:
            nu_sweep_tmp = nu_sweep_norm.iloc[:, iS][mut]
            mu_tmp = interpolate_slope(mu[mut], muNS[mut], nu_sweep_tmp)

        xfit = np.linspace(0, x.max())
        yfit = fun(xfit, mu_tmp, s.iloc[iS])
        ax.plot(xfit, yfit, lw=2, color=colors[iS], alpha=0.5)

    # Costs
    ax = axs[1]
    x = np.array(s.index)
    y = np.array(s)
    if ds is not None:
        dy = np.array(ds)

    imin = y.argmax()
    if ds is not None:
        ymin = 1.1 * (y[imin] + dy[imin])
    else:
        ymin = 1.1 * y[imin]

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
                [s.iloc[0], s.iloc[1]],
                lw=2,
                ls='--',
                color='k',
               )
        ax.annotate('Full conservation',
                    xy=(1.1e-3, 0.9 * s.iloc[0]),
                    xytext=(1.1e-3, 0.01 * s.iloc[0]),
                    arrowprops={'facecolor': 'black',
                                'width': 1.5,
                                'headlength': 10,
                                'shrink': 0.1,
                               },
                    ha='left',
                    va='center',
                    fontsize=fs,
                   )

    ax.set_xlabel('Variability in subtype B [bits]', fontsize=fs)
    ax.set_ylabel('Fitness cost', fontsize=fs)
    ax.set_xlim(0.9e-3, 1.1)
    ax.set_ylim(1e-5, max(0.1, ymin))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_tick_params(labelsize=fs)
    ax.yaxis.set_tick_params(labelsize=fs)

    plt.tight_layout()


    if savefig:
        fig_filename = savefig
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
                n_templates = p.n_templates_dilutions[ind]

            # Get site entropy
            if pos not in comap.index:
                continue
            pos_ref = comap.loc[pos]
            S_pos = ref.entropy[pos_ref]

            # Keep only sites where the ancestral allele and group M agree
            if ref.consensus_indices[pos_ref] != aft_pos[:, 0].argmax():
                continue

            # Filter out sweeps if so specified
            if no_sweeps:
                found = False
                nuc_anc = p.initial_sequence[pos]
                for ia, aft_nuc in enumerate(aft_pos[:4]):
                    if (alpha[ia] != nuc_anc) and (aft_nuc > 0.5).any():
                        found = True
                if found:
                    continue

            for ia, aft_nuc in enumerate(aft_pos[:4]):
                # Keep only derived alleles
                if alpha[ia] == p.initial_sequence[pos]:
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

                for it, (t, af_nuc, n_temp) in enumerate(izip(times, aft_nuc, n_templates)):
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
                             'n_templates': n_temp,
                            }
                    data.append(datum)

    data = pd.DataFrame(data)

    return data



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

    # Make time and entropy bins
    t_bins = np.array([0, 500, 1000, 1500, 2000, 3000], int)
    t_binc = 0.5 * (t_bins[:-1] + t_bins[1:])
    add_binned_column(data, t_bins, 'time')
    data['time_binc'] = t_binc[data['time_bin']]

    # No-entropy sites are many, so the bin 0 comes up twice...
    S_bins = np.percentile(data['S'], np.linspace(0, 100, 8))[1:]
    S_binc = 0.5 * (S_bins[:-1] + S_bins[1:])
    n_alleles = np.array(data.loc[:, ['af', 'S_bin']].groupby('S_bin').count()['af'])
    add_binned_column(data, S_bins, 'S')
    data['S_binc'] = S_binc[data['S_bin']]

    fnS = fn.split('.')[0]+'_Sbins.npz'
    np.savez(fnS, bins=S_bins, binc=S_binc, n_alleles=n_alleles)

    # Estimate initial slope from data themselves, without sweeps
    muNS = fit_mu_highentropy(data)

    # Estimate the mutation rate from the high-entropy class without sweeps
    alpha = compare_mu(muNS, plot=True)

    #sys.exit()

    # Load the fraction of sweeps for entropy and mutation classes, normalized
    # by the high-entropy value (the one we use to calculate alpha)
    fn_sw = 'data/fraction_sweep_entropy_mut.pickle'
    nu_sweep = pd.read_pickle(fn_sw)
    nu_sweep_norm = (nu_sweep.T / nu_sweep.T.iloc[-1]).T

    # Set mutation rates
    tmp = load_mutation_rates()
    mu = tmp['mu']
    muA = tmp['muA']

    # Fitness estimate
    data_to_fit = prepare_data_for_fit(data, plot=True)
    s = fit_fitness_cost_interpmu(data_to_fit,
                                  mu=mu,
                                  muNS=muNS,
                                  nu_sweep_norm=nu_sweep_norm)

    #sys.exit()

    if True:
        def fit_fitness_cost_for_bootstrap(data):
            data_to_fit = prepare_data_for_fit(data, plot=False)
            s =  fit_fitness_cost_interpmu(data_to_fit,
                                           mu=mu,
                                           muNS=muNS,
                                           nu_sweep_norm=nu_sweep_norm)
            return s['s']

        ds = s['s'].copy()
        sBS = boot_strap_patients(data, fit_fitness_cost_for_bootstrap, n_bootstrap=100)
        for key, _ in ds.iteritems():
            ds[key] = np.std([tmp[key] for tmp in sBS])
        s.rename(columns={'ds': 'ds_fit'}, inplace=True)
        s['ds_bootstrap'] = ds
        s.sort_index(axis=1, ascending=False, inplace=True)

    fn_s = 'data/fitness_cost_result.pickle'
    s.to_pickle(fn_s)

    for mut in ['A->G', 'G->A', 'C->T', 'T->C']:
        plot_fitness_cost(data_to_fit,
                          s['s'], mu, ds=s['ds_bootstrap'],
                          muNS=muNS,
                          mut=mut,
                          nu_sweep_norm=nu_sweep_norm,
                          savefig='figures/fitness_cost_saturation_'+mut,
                         )
