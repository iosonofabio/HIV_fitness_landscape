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
    '''
    Calculate the mutation rate matrix from accumulatio of
    intra patient diversity via linear regression. Uncertainty
    of the estimates is assessed via boot strapping over patients.
    '''
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

    # bin data into time bins and extract derived allele frequency
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

    if axs is None: # create new axis and save plot
        savefig = True
        fig, axs = plt.subplots(1,2, figsize=(12,6))
    else:           # add to provided axis
        savefig = False

    mlist = ['A->G', 'C->T', 'G->A', 'T->C',
             'A->C', 'A->T', 'C->A', 'C->G',
             'G->T', 'G->C', 'T->G', 'T->A']
    for mut in mlist:
        aft = d.loc[mut]
        if mut in transitions: # transitions in one plot
            ax=axs[0]
            color = cmap[transitions.index(mut)]
            marker='o'
            ls='-o'
        else:                   # transversion in the other plot
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

    yticks = [0.0, 0.01, 0.02, 0.03, 0.04]
    axs[0].set_yticks(yticks)

    yticks = [0.0, 0.003, 0.006, 0.009, 0.012]
    axs[1].set_yticks(yticks)

    plt.tight_layout()

    if savefig:
        plt.savefig('mutation_linear_increase.png')


def plot_mutation_rate_graph(mu, ax=None):
    '''Plot the figure that illustrates all mutations as arrows between nucleotides'''
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    else:
        fig=None

    fs = 16
    lim = 6.7
    ax.set_xlim(-lim, lim * 2 + 10 + 0.3)
    ax.set_ylim(lim, -lim)
    ax.axis('off')

    nucs = ['A', 'C', 'G', 'T']
    rc = 4
    r = 1.4
    xoff = -1
    for iy, yc in enumerate([-rc, rc]):
        for ix, xc in enumerate([-rc, rc]):
            circ = plt.Circle((xc + xoff, yc), radius=r,
                              edgecolor='black',
                              facecolor=([0.9] * 3),
                              lw=2.5,
                             )
            ax.add_patch(circ)
            i = 2 * iy + ix
            ax.text(xc + xoff, yc, nucs[i], ha='center', va='center', fontsize=34)

    def get_arrow_properties(mut, scale=1.0):
        from matplotlib import cm
        cmap = cm.jet
        wmin = 0.2
        wmax = 0.6
        fun = lambda x: np.log10(x)
        mumin = fun(1e-7)
        mumax = fun(2e-5)
        if isinstance(mut, basestring):
            m = fun(mu.loc[mut, 'mu'])
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
    ax.arrow(xoff -(rc + gap), -(rc - r - 0.2), 0, 2 * (rc - r - 0.2), length_includes_head=True, **get_arrow_properties('A->G'))
    ax.arrow(xoff -(rc - gap), (rc - r - 0.2), 0, -2 * (rc - r - 0.2), length_includes_head=True, **get_arrow_properties('G->A'))
    ax.arrow(xoff +(rc - gap), -(rc - r - 0.2), 0, 2 * (rc - r - 0.2), length_includes_head=True, **get_arrow_properties('C->T'))
    ax.arrow(xoff +(rc + gap), +(rc - r - 0.2), 0, -2 * (rc - r - 0.2), length_includes_head=True, **get_arrow_properties('T->C'))
    ax.arrow(xoff -(rc - r - 0.2), -(rc + gap), 2 * (rc - r - 0.2), 0, length_includes_head=True, **get_arrow_properties('A->C'))
    ax.arrow(xoff +(rc - r - 0.2), -(rc - gap), -2 * (rc - r - 0.2), 0, length_includes_head=True, **get_arrow_properties('C->A'))
    ax.arrow(xoff -(rc - r - 0.2), +(rc - gap), 2 * (rc - r - 0.2), 0, length_includes_head=True, **get_arrow_properties('G->T'))
    ax.arrow(xoff +(rc - r - 0.2), +(rc + gap), -2 * (rc - r - 0.2), 0, length_includes_head=True, **get_arrow_properties('T->G'))
    ax.arrow(xoff -(rc - r - 0.7), -(rc - r - 0.2), 2 * (rc - r - 0.4), 2 * (rc - r - 0.4), length_includes_head=True, **get_arrow_properties('A->T'))
    ax.arrow(xoff +(rc - r - 0.7), +(rc - r - 0.2), -2 * (rc - r - 0.4), -2 * (rc - r - 0.4), length_includes_head=True, **get_arrow_properties('T->A'))
    ax.arrow(xoff -(rc - r - 0.5), +(rc - r - 0.2), 2 * (rc - r - 0.3), -2 * (rc - r - 0.3), length_includes_head=True, **get_arrow_properties('G->C'))
    ax.arrow(xoff +(rc - r - 0.5), -(rc - r - 0.2), -2 * (rc - r - 0.3), +2 * (rc - r - 0.3), length_includes_head=True, **get_arrow_properties('C->G'))


    oft = 0.8
    ax.text(rc + 6.4, - 5 + oft - 1.7, 'Rates [per site per day]:', fontsize=fs)

    def write_mut(mut, dy, dx):
        decade = int(np.floor(np.log10(mu.loc[mut, 'mu'])))
        flo = mu.loc[mut, 'mu'] * 10**(-decade)
        dflo = mu.loc[mut, 'dmulog10'] *  np.log(10) * mu.loc[mut, 'mu'] * 10**(-decade)
        if dflo >= 0.5:
            flot = '{:1.0f}'.format(np.round(flo, 0))
            dflot = '{:1.0f}'.format(np.round(dflo, 0))
        else:
            flot = '{:1.1f}'.format(np.round(flo, 1))
            dflot = '{:1.0f}'.format(np.round(10 * dflo, 0))
        mtxt = '$' + flot + '('+ dflot+')'+ ' \cdot ' + '10^{'+str(decade)+'}$'
        ax.text(4 + 2.0 + dx, -5 + oft - 0.6 + dy, mut[0]+u' \u2192 '+mut[-1], fontsize=fs)
        ax.arrow(4 + 4.4 + dx, - 5 + oft - 0.8 + dy, 1.8, 0, length_includes_head=True,
                 **get_arrow_properties(mu.loc[mut, 'mu'], scale=0.7))
        ax.text(4 + 10.2 + dx, -5 + oft - 0.6 + dy, mtxt, ha='right', fontsize=fs)

    dy = 0.5
    dx = 0
    muts = ['G->C', 'A->T', 'C->G', 'A->C', 'G->T', 'T->A',
            'T->G', 'C->A', 'A->G', 'T->C', 'C->T', 'G->A']
    for imut, mut in enumerate(muts):
        write_mut(mut, dy, dx)
        dy += 2
        if imut + 1 == len(muts) // 2:
            dy -= len(muts) // 2 * 2
            dx += 9

    return fig, ax


def plot_figure_1(data, mu, dmulog10, muA, dmuAlog10,suffix=''):
    '''Plot figure 1 of the paper'''
    print('Plot Figure 1')
    fig = plt.figure(figsize=(12, 11))
    ax1 = plt.subplot2grid((2,2), (0,0))
    ax2 = plt.subplot2grid((2,2), (0,1))
    ax3 = plt.subplot2grid((2,2), (1, 0), colspan=2)
    # plot linear regression
    plot_mutation_increase(data, mu=mu, axs=[ax1, ax2])

    mu_all = pd.DataFrame({'mu': mu,
                              'muA': muA,
                              'dmulog10': dmulog10,
                              'dmuAlog10': dmuAlog10,
                            })
    # plot matrix of arrows
    plot_mutation_rate_graph(mu_all,
                             ax=ax3)
    plt.tight_layout()

    # Add labels
    from util import add_panel_label
    add_panel_label(ax1, 'A', x_offset=-0.2)
    add_panel_label(ax2, 'B', x_offset=-0.2)
    add_panel_label(ax3, 'C', x_offset=-0.08)

    plt.ion()
    plt.show()


    for ext in ['svg', 'png', 'pdf']:
        fig.savefig('../figures/figure_1'+suffix+'.'+ext)


def export_mutation_rate_matrix(mu, dmulog10, muA=None, dmuAlog10=None, suffix=''):
    '''Export the table of mutation rate coefficients to file'''
    fn = '../data/mutation_rates/mutation_rate'+suffix+'.pickle'
    out = {'mu': mu, 'dmulog10': dmulog10}
    if muA is not None:
        out['muA'] = muA
    if dmuAlog10 is not None:
        out['dmuAlog10'] = dmuAlog10

    mu_out = pd.DataFrame(out)
    mu_out.to_pickle(fn)

    fn_tsv = '../data/mutation_rates/mutation_rate'+suffix+'.tsv'
    mu_out['log10mu'] = np.log10(out['mu'])
    mu_out['dlog10mu'] = out['dmulog10']
    header = ['log10(mu [per day per site])', 'stddev(log10(mu [per day per site]))']
    mu_out[['log10mu', 'dlog10mu']].to_csv(fn_tsv,
                                        sep='\t',
                                        header=header,
                                        float_format='%1.2f')


def collect_data(patients, cov_min=100, refname='HXB2', subtype='any',
                 entropy_threshold=0.1, excluded_proteins=[]):
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
            # skip if protein is to be excluded
            if fead['protein_codon'][0][0] in excluded_proteins:
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
                if S_pos < entropy_threshold:
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
                             'refpos': comap.loc[pos],
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
    # use only patients with early samples and likely single virion infection
    patients = ['p1', 'p2','p5', 'p6', 'p8', 'p9', 'p11']
    data_out_path = '../data/mutation_rates/'

    # make many mutation rate estimates excluding gp120 or not and with different
    # threshold for cross-sectional diversity
    for excluded_proteins in [[], ['gp120']]:
        for thres in [0.01, 0.03, 0.1, 0.3, 0.5]:
            suffix = '_'+'_'.join([str(thres)]+excluded_proteins)

            # Intermediate data are saved to file for faster access later on
            fn = data_out_path + 'mutation_rate_data'+suffix+'.pickle'
            if not os.path.isfile(fn) or args.regenerate:
                data = collect_data(patients, entropy_threshold=thres, excluded_proteins=excluded_proteins)
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

            # save positions used for mutation rate estimation.
            all_pos = np.array(np.unique(data['refpos']), dtype=int)
            print(thres, excluded_proteins, '# positions', len(all_pos))
            np.savetxt(data_out_path+ 'mutation_rate_positions'+suffix+'.txt', all_pos, fmt='%d')

            # Get mutation rate with bootstrap error bars
            mu,dmulog10 = get_mutation_matrix(data)

            # Compare to Abram et al 2010
            tmp = get_mu_Abram2010(with_std=True)
            muA = tmp['mu']
            dmuAlog10 = tmp['std'] / tmp['mu'] / np.log(10)

            # Save results to file (used in Figure 2)
            export_mutation_rate_matrix(mu, dmulog10, muA, dmuAlog10, suffix=suffix)

            # Plot Figure 1
            if 'gp120' in excluded_proteins and thres==0.3:
                plot_figure_1(data=data, mu=mu, dmulog10=dmulog10, muA=muA, dmuAlog10=dmuAlog10, suffix=suffix)
