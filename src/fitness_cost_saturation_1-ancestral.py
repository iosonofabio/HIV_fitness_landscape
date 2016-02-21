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


def fit_mu_highentropy(data):
    '''An estimate of the high-entropy initial slope is necessary since we exclude sweeps'''
    from mutation_rate import get_mutation_matrix, plot_comparison

    data_mu = data.loc[data['S_binc'] ==  data['S_binc'].max()].copy()
    mu = get_mutation_matrix(data_mu)
    mu.name = 'mutation rate, from high-entropy sites'
    return mu


def prepare_data_for_fit(data, plot=False):
    '''Prepare the data for fit by averaging single allele frequencies'''
    def plot_bins(counter):
        fs = 14
        fig, ax = plt.subplots()
        h, bins = np.histogram(counter, bins=10)
        for i in xrange(len(h)):
            ax.bar(bins[i], h[i], bins[i+1] - bins[i])
        ax.set_xlabel('number of allele in each frequency average', fontsize=fs)
        ax.set_xlabel('number of averages', fontsize=fs)
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
        ax.xaxis.set_tick_params(labelsize=fs)
        ax.yaxis.set_tick_params(labelsize=fs)
        ax.text(0.98, 0.95,
                'n. averages = 5 times x 6 entropies x 12 muts',
                ha='right',
                va='top',
                transform=ax.transAxes)

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


def plot_fit(data_to_fit, mu, s):
    from matplotlib import cm
    fig_width = 5
    fs = 16
    fig, axs = plt.subplots(1, 2,
                            figsize=(2 * fig_width, fig_width))

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
                color='k',
                label='saturation', marker='o',
                markersize=10,
               )

#    ax.plot([1e-3, s.index[1]],
#            [s['s'].iloc[0], s['s'].iloc[1]],
#            lw=2,
#            ls='--',
#            color='k',
#           )
#    ax.errorbar([1e-3], [s['s'].iloc[0]],
#                yerr=[s['ds'].iloc[0]],
#                lw=2,
#                color='k'
#               )

    ## include estimates from pooled allele frequency fits
    with open('data/combined_af_avg_selection_coeff.pkl', 'r') as f:
        import cPickle as pickle
        caf_s = pickle.load(f)

    ax.errorbar(caf_s['all'][:,0], caf_s['all'][:,1],
                yerr=caf_s['all_std'][:,1], marker='o',
                lw=2,
                color='r',
                label='pooled',
                markersize=10,
#                ls='-o'
               )

    ## include estimates from KL fits
    x = np.loadtxt('figures/Vadim/entropy_quantile.txt')
    y = np.median(np.loadtxt('figures/Vadim/smuD_KL.txt')[:,:-2], axis=0)

    ax.plot(x[1:],y[1:], '-o',lw=2,
                color='g',
                label='KL',
                markersize=10,
#                ls='-o'
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

    ax.set_xlabel('Variability in group M [bits]', fontsize=fs)
    ax.set_ylabel('Fitness cost', fontsize=fs)
    ax.set_xlim(0.9e-3, 1.1)
    ax.set_ylim(9e-5, 0.11)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_tick_params(labelsize=fs)
    ax.yaxis.set_tick_params(labelsize=fs)

    plt.tight_layout()
    plt.legend(loc=3, fontsize=fs)
    plt.ion()
    plt.show()


def fit_fitness_cost_simplest(data, plot=True, bootstrap=True, mu=None):
    '''Fit one slope and 6 saturations to ALL data at once'''
    def average_data(data):
        data = data.copy()
        data['counter'] = 1.0
        d = (data
             .loc[:, ['S_binc', 'time_binc', 'af', 'counter']]
             .groupby(['S_binc', 'time_binc'], as_index=True)
             .sum())
        d['af'] /= d['counter']
        data_to_fit = d['af'].unstack()
        data_to_fit.name = 'af'
        return data_to_fit

    def fit_data(data_to_fit, mu=None):
        from scipy.optimize import curve_fit

        # First fit slope from high-entropy class
        if mu is None:
            datum = data_to_fit.iloc[-1]
            nInd = 3
            x = np.array(datum.index)[:3]
            y = np.array(datum)[:3]
            mu = np.dot(x, y) / np.dot(x, x)

        # Then, fit the saturations
        s = []
        fun = lambda x, s: mu / s * (1.0 - np.exp(-s * x))
        for iS, (S, datum) in enumerate(data_to_fit.iterrows()):
            x = np.array(datum.index)
            y = np.array(datum)
            s0 = 1e-2 / 10**(iS / 2.0)
            fit = curve_fit(fun, x, y, p0=[s0])
            sTmp = fit[0][0]
            dsTmp = fit[1][0][0]
            s.append({'S': S,
                      's': sTmp,
                      'ds': dsTmp,
                     })
        s = pd.DataFrame(s).set_index('S', drop=True, inplace=False)
        return mu, s

    data_to_fit = average_data(data)
    mu, s = fit_data(data_to_fit, mu=mu)

    if bootstrap:
        def bootstrap_fun():
            def prepare_and_fit(data):
                data_to_fit = average_data(data)
                return fit_data(data_to_fit, mu=mu)[1]['s']

            ds = s['s'].copy()
            sBS = boot_strap_patients(data, prepare_and_fit, n_bootstrap=100)
            for key, _ in ds.iteritems():
                ds[key] = np.std([tmp[key] for tmp in sBS])
            s['ds'] = ds

        bootstrap_fun()

    if plot:
        plot_fit(data_to_fit, mu, s)

    return {'data_to_fit': data_to_fit,
            'mu': mu,
            's': s}


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

            # Keep only 1 - ancestral allele
            ia = p.initial_indices[pos]
            aft_nuc = 1 - aft_pos[ia]
            for it, (t, af_nuc, n_temp) in enumerate(izip(times, aft_nuc, n_templates)):
                datum = {'time': t,
                         'af': af_nuc,
                         'pos': pos,
                         'pos_ref': pos_ref,
                         'protein': fead['protein_codon'][0][0],
                         'pcode': pcode,
                         'ancestral': alpha[ia],
                         'S': S_pos,
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

    fn = 'data/fitness_cost_data_1-ancestral.pickle'
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
    perc = np.linspace(0, 100, 8)
    S_bins = np.percentile(data['S'], perc)[1:]
    #S_binc = 0.5 * (S_bins[:-1] + S_bins[1:])
    S_binc = np.percentile(data['S'], 0.5*(perc[:-1]+perc[1:]))[1:]
    n_alleles = np.array(data.loc[:, ['af', 'S_bin']].groupby('S_bin').count()['af'])
    add_binned_column(data, S_bins, 'S')
    data['S_binc'] = S_binc[data['S_bin']]

    fnS = fn.split('.')[0]+'_Sbins.npz'
    np.savez(fnS, bins=S_bins, binc=S_binc, n_alleles=n_alleles)

    # Simplest model, dump all together no matter what the mutation rate
    a = fit_fitness_cost_simplest(data, mu=1.19e-5)
    mu = a['mu']
    s = a['s']
    data_to_fit = a['data_to_fit']
