# vim: fdm=indent
'''
author:     Fabio Zanini
date:       15/06/15
content:    Figure out what mutations are preferred for sweeps.
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


def genome_fractions():
    # TODO: this should be real nonsyn opportunities rather than just genome fraction
    from collections import Counter
    ref = HIVreference(subtype='B', load_alignment=False)
    d = pd.Series(Counter(ref.consensus)).loc[alpha[:4]]
    return d / d.sum()


def nonsyn_chances():
    from Bio.Seq import Seq
    chances = {a+'->'+b: 0 for a in alpha[:4] for b in alpha[:4] if a != b}
    ref = HIVreference(refname='NL4-3', load_alignment=True)
    seq = ref.seq
    for _, anno in ref.annotation.iteritems():
        if anno.type != 'gene':
            continue
        seq_reg = anno.extract(seq).seq
        for icod in xrange(len(seq_reg) // 3):
            cod_anc = seq_reg[icod * 3: (icod + 1) * 3]
            aa_anc = cod_anc.translate()
            for rf in xrange(3):
                for nuc in alpha[:4]:
                    if nuc == cod_anc[rf]:
                        continue
                    aa_der = np.array(seq_reg[icod * 3: (icod + 1) * 3])
                    aa_der[rf] = nuc
                    aa_der = Seq(''.join(aa_der)).translate()
                    if aa_anc != aa_der:
                        chances[cod_anc[rf]+'->'+nuc] += 1

    chances = pd.Series(chances)
    chances /= chances.sum()
    return chances


def collect_data(patients, cov_min=100, refname='HXB2'):
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

                # Keep only sweeps
                if not (aft_nuc > 0.5).any():
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


def plot_sweeps(data):
    '''Plot the sweeps and have a look'''
    
    fig, ax = plt.subplots()
    for (pcode, pos, mut), datum in data.groupby(['pcode', 'pos', 'mut']):
        x = np.array(datum['time'])
        y = np.array(datum['af'])
        ax.plot(x, y, lw=2)

    ax.set_xlabel('Time [days since EDI]')
    ax.set_ylabel('$\\nu$')

    plt.ion()
    plt.show()


def plot_fraction_sweeps_entropy_mut(nu_sweep):
    from matplotlib import cm
    cmap = cm.jet

    fig, ax = plt.subplots()
    for i, (mut, tmp) in enumerate(nu_sweep.iterrows()):
        x = np.array(tmp.index)
        y = np.array(tmp)
        color = cm.jet(1.0 * i / nu_sweep.shape[0])
        ax.plot(x, y, label=mut, lw=2, color=color)

    for xtmp in x:
        ax.annotate('', xy=(xtmp, 0.9), xytext=(xtmp, 0.99),
                    arrowprops=dict(facecolor='black', shrink=0.05))

    ax.legend(bbox_to_anchor=(0.32, 0.8),
              bbox_transform=ax.transAxes,
              ncol=2)
    ax.set_xlabel('Sbinc')
    ax.set_ylabel('Fraction of sweeps')
    ax.set_xscale('log')
    ax.set_ylim(0, 1)
    ax.set_xlim(0.9 * x.min(), 1.1 * x.max())

    plt.ion()
    plt.show()



# Script
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sweeps and mutations')
    parser.add_argument('--regenerate', action='store_true',
                        help="regenerate data")
    args = parser.parse_args()

    fn = 'data/sweeps_mutations_data.pickle'
    if not os.path.isfile(fn) or args.regenerate:
        patients = ['p1', 'p2', 'p3','p5', 'p6', 'p8', 'p9', 'p11']
        cov_min = 100
        data = collect_data(patients, cov_min=cov_min)
        data.to_pickle(fn)
    else:
        data = pd.read_pickle(fn)

    # Make time and entropy bins
    t_bins = np.array([0, 500, 1000, 1500, 2000, 3000], int)
    t_binc = 0.5 * (t_bins[:-1] + t_bins[1:])
    add_binned_column(data, t_bins, 'time')
    data['time_binc'] = t_binc[data['time_bin']]

    # Entropy binning based on quantiles of ALL alleles (not only sweeps!)
    fnS = 'data/fitness_cost_data_nosweep_Sbins.npz'
    S_bins = np.load(fnS)['bins']
    S_binc = np.load(fnS)['binc']
    add_binned_column(data, S_bins, 'S')
    data['S_binc'] = S_binc[data['S_bin']]

    # Number alleles in each entropy bin for each mutation, to adjust the initial
    # slope of the fitness cost fits
    nu_sweep = data.loc[:, ['pos', 'pcode', 'af', 'S_binc', 'mut']].groupby(['mut', 'S_binc']).count()['af'].unstack().fillna(0)
    # Pseudocounts
    nu_sweep += 1
    # Normalize
    nu_sweep = (nu_sweep.T / nu_sweep.T.sum(axis=0)).T
    fn_sw = 'data/fraction_sweep_entropy_mut.pickle'
    nu_sweep.to_pickle(fn_sw)

    plot_fraction_sweeps_entropy_mut(nu_sweep)

    sys.exit()

    # Group single trajectories
    nMu = (data.loc[data['syn'] == False]
           .groupby(['pcode', 'pos', 'mut'])
           .count()['af']
           .unstack()
           .fillna(0)
           .sum(axis=0))

    mu = load_mutation_rates()['mu']
    frac = nonsyn_chances()

    sys.exit()

    # The null expectation is that we have frac(A) * mu(A->G) / sum() sweeps
    nNull = frac * mu
    nNull /= nNull.sum()

    print 'Overrepresentation of mutations in nonsyn sweeps compared to frac * mu:'
    print np.round(1 * nMu / nMu.sum() / nNull, 2)

    # Divided by patient
    nMuP = (data.loc[data['syn'] == False]
            .groupby(['pos', 'mut', 'pcode'])
            .count()['af']
            .unstack(['mut', 'pcode'])
            .fillna(0)
            .sum(axis=0)
            .unstack()
            .fillna(0))

    print 'Substitutions in nonsynonymous sweeps by patient [%]:'
    print np.round(nMuP / nMuP.sum(axis=0) * 100, 0)

    #plot_sweeps(data)
