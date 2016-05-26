# vim: fdm=indent
'''
author:     Fabio Zanini
date:       15/06/15
content:    Characterize sweeps a bit.
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
def collect_data(patients, cov_min=100, refname='HXB2'):
    '''Collect data for the fitness cost estimate'''
    ref = HIVreference(refname=refname, subtype='any', load_alignment=True)

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
                anc_cross = False
            else:
                anc_cross = True

            for ia, aft_nuc in enumerate(aft_pos[:4]):
                # Keep only derived alleles
                if alpha[ia] == p.initial_sequence[pos]:
                    continue

                # Keep only sweeps
                if not (aft_nuc > 0.9).any():
                    continue

                # Annotate with syn/nonsyn alleles
                cod_new = cod_anc[:pc] + alpha[ia] + cod_anc[pc+1:]
                if translate(cod_anc) != translate(cod_new):
                    syn = False
                else:
                    syn = True

                mut = p.initial_sequence[pos]+'->'+alpha[ia]
                for it, (t, af_nuc) in enumerate(izip(times, aft_nuc)):
                    datum = {'time': t,
                             'af': af_nuc,
                             'pos': pos,
                             'pos_ref': pos_ref,
                             'protein': fead['protein_codon'][0][0],
                             'pcode': pcode,
                             'mut': mut,
                             'S': S_pos,
                             'syn': syn,
                             'anc_cross': anc_cross,
                            }
                    data.append(datum)

    data = pd.DataFrame(data)

    return data


def plot_sweeps(data):
    '''Plot the sweeps and have a look'''
    ref = HIVreference(refname='HXB2', subtype='B', load_alignment=False)
    pos_genes = {genename: list(ref.annotation[genename].location)
                 for genename in ['gag', 'pol', 'env']}
    palette = sns.color_palette("husl", 8)
    colors = {'gag': palette[1],
              'pol': palette[0],
              'env': palette[4],
              'other': 'grey'}
    zs = {'gag': 1,
          'pol': 2,
          'env': 0,
          'other': 3}
    def get_color(pos_ref):
        '''Get color of line based on the gene'''
        for genename, pos_gene in pos_genes.iteritems():
            if pos_ref in pos_gene:
                return colors[genename]
        return colors['other']

    def get_z(pos_ref):
        '''Get z (depth) of line based on the gene'''
        for genename, pos_gene in pos_genes.iteritems():
            if pos_ref in pos_gene:
                return colors[genename]
        return colors['other']
    
    fig, axs = plt.subplots(3, 3, figsize=(9, 9))
    axs = axs.ravel()
    pcodes = sorted(data['pcode'].unique(), key=lambda x: int(x[1:]))
    for (pcode, data_pat) in data.groupby('pcode'):
        ip = pcodes.index(pcode)
        ax = axs[ip]
        for (pos_ref, mut), datum in data_pat.groupby(['pos_ref', 'mut']):
            x = np.array(datum['time'])
            x += 100. * pos_ref / 9000
            y = np.array(datum['af'])
            ax.plot(x, y, lw=2, zorder=get_z(pos_ref), color=get_color(pos_ref), alpha=0.3)
        ax.set_ylim(0.001, 0.999)
        ax.set_title(pcode)
        if ip > 5:
            ax.set_xlabel('Time [days since EDI]')
        if (ip % 3) == 0:
            ax.set_ylabel('Frequency')
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
    ax = axs[-1]
    for genename in ['gag', 'pol', 'env']:#, 'other']:
        ax.plot([0], color=colors[genename], label=genename)
    ax.legend(loc='center', fontsize=16)
    ax.set_axis_off()

    plt.tight_layout()

    plt.ion()
    plt.show()

    return fig


def quantify_beneficial(data, syn='any'):
    '''Quantify the beneficial effect based on frequency trajectories'''
    sw = []

    if syn != 'any':
        data = data.groupby('syn').get_group(syn)

    pcodes = sorted(data['pcode'].unique(), key=lambda x: int(x[1:]))
    for (pcode, data_pat) in data.groupby('pcode'):
        ip = pcodes.index(pcode)
        for (pos_ref, mut), datum in data_pat.groupby(['pos_ref', 'mut']):
            x = np.array(datum['time'])
            y = np.array(datum['af'])

            if len(x) == 1:
                continue
            dy = np.diff(y)
            dx = np.diff(x)
            d = dy / dx
            am = d.argmax()
            sw.append({'pcode': pcode,
                       'dy/dx': d[am],
                       'dy': dy[am],
                       'dx': dx[am],
                       'x0': x[am],
                       'y0': y[am],
                       'x1': x[am+1],
                       'y1': y[am+1],
                      })
    sw = pd.DataFrame(sw)

    dxs = [10, 60, 150, 250, 350, 550, 800]
    dys = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    m = []
    for i in xrange(len(dxs) - 1):
        swi = sw.loc[(sw['dx'] >= dxs[i]) & (sw['dx'] < dxs[i+1])]
        for dy in dys:
            frac = (swi['dy'] >= dy).mean()
            num = (swi['dy'] >= dy).sum()
            m.append({'dx_min': dxs[i],
                      'dx_max': dxs[i+1],
                      'dx': 0.5 * (dxs[i] + dxs[i+1]),
                      'dy': dy,
                      'frac': frac,
                      'num': num,
                     })
    m = pd.DataFrame(m)

    return m


def plot_fraction_sweep(m):
    '''Plot the fraction of sweeps based on the dx-dy criteria'''
    fig, ax = plt.subplots(figsize=(6, 4))
    a = m[['dx', 'dy', 'frac']].groupby(['dx', 'dy']).sum().unstack()['frac']

    #sns.heatmap(a, vmin=0, vmax=1)

    palette = np.array(sns.color_palette("husl", 8))[[1, 0, 4, 6, 2, 5]]
    for idx, (dx, datum) in enumerate(a.iterrows()):
        x = np.array(datum.index)
        y = np.array(datum)
        if dx == 300:
            lw = 2.3
            zorder = 4
            color = 'red'
        else:
            lw = 2
            zorder = 3
            color = palette[idx]
        ax.plot(x, y, lw=lw, color=color, zorder=zorder, label=str(int(dx)))
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Fraction dy > threshold')
    ax.legend(loc=3, ncol=2, title='Interval [days]')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    # Bar for fast sweeps
    ax.vlines(0.96, 0.005, 0.10, lw=2, color='k', zorder=5)
    ax.hlines(0.005, 0.94, 0.98, lw=2, color='k', zorder=5)
    ax.hlines(0.10, 0.94, 0.98, lw=2, color='k', zorder=5)
    ax.text(0.98, 0.035, '*', fontsize=20, ha='center', va='center')

    # Bar for slow sweeps
    ax.vlines(0.38, 0.9, 0.995, lw=2, color='k', zorder=5)
    ax.hlines(0.9, 0.36, 0.40, lw=2, color='k', zorder=5)
    ax.hlines(0.995, 0.36, 0.40, lw=2, color='k', zorder=5)
    ax.text(0.41, 0.94, '+', fontsize=18, ha='center', va='center')

    plt.tight_layout()
    plt.ion()
    plt.show()

    return fig




# Script
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Characterize sweeps')
    parser.add_argument('--regenerate', action='store_true',
                        help="regenerate data")
    args = parser.parse_args()

    fn = 'data/sweeps.pickle'
    if not os.path.isfile(fn) or args.regenerate:
        patients = ['p1', 'p2', 'p3','p5', 'p6', 'p8', 'p9', 'p11']
        cov_min = 100
        data = collect_data(patients, cov_min=cov_min)
        data.to_pickle(fn)
    else:
        data = pd.read_pickle(fn)

    #fig1 = plot_sweeps(data)
    #for ext in ['svg', 'pdf', 'png']:
    #    fig1.savefig('figures/sweeps_lines.'+ext)

    for syn in ['any', False, True]:
        sw = quantify_beneficial(data, syn=syn)
        fig2 = plot_fraction_sweep(sw)
        if syn == True:
            fig2.suptitle('Synonymous only')
            for ext in ['svg', 'pdf', 'png']:
                fig2.savefig('figures/sweeps_quantification_syn.'+ext)
        elif syn == False:
            fig2.suptitle('Nonsynonymous only')
            for ext in ['svg', 'pdf', 'png']:
                fig2.savefig('figures/sweeps_quantification_nonsyn.'+ext)
        else:
            for ext in ['svg', 'pdf', 'png']:
                fig2.savefig('figures/sweeps_quantification.'+ext)

