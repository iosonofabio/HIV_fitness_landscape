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
from secondary_structure_test import load_secondary_structure_patient



# Functions
def load_mutation_rates():
    fn = 'data/mutation_rate.pickle'
    return pd.read_pickle(fn)


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
        p.sec_str = load_secondary_structure_patient(pcode)

        comap = (pd.DataFrame(p.map_to_external_reference('genomewide', refname=refname)[:, :2],
                              columns=[refname, 'patient'])
                   .set_index('patient', drop=True)
                   .loc[:, refname])

        aft = p.get_allele_frequency_trajectories('genomewide', cov_min=cov_min)
        for pos, aft_pos in enumerate(aft.swapaxes(0, 2)):
            fead = p.pos_to_feature[pos]

            # Look at the protein secondary structure
            sec_str = p.sec_str[pos]

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
                             'protein_secondary_structure': sec_str,
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

    fn = 'data/fitness_cost_protein_secondary_structure.pickle'
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

    # Average by time bin, syn, and sec structure
    dav = (data
           .loc[:, ['time_binc', 'syn', 'protein_secondary_structure', 'af']]
           .groupby(['time_binc', 'syn', 'protein_secondary_structure'])
           .mean()
           .loc[:, 'af']
           .unstack('time_binc'))

    def average_data(data):
        dav = (data
               .copy()
               .loc[:, ['time_binc', 'syn', 'protein_secondary_structure', 'af']]
               .groupby(['time_binc', 'syn', 'protein_secondary_structure'])
               .mean()
               .loc[:, 'af']
               #.unstack('time_binc')
              )
        return dav
    from util import boot_strap_patients
    reps = pd.concat(boot_strap_patients(data, average_data, n_bootstrap=100),
                     axis=1)
    reps.columns = np.arange(reps.shape[1]) + 1

    dav = pd.concat([reps.mean(axis=1), reps.std(axis=1)], axis=1)
    dav.columns = ['mean', 'std']

    def plot_average_frequencies(dav):
        fig, ax = plt.subplots()
        fs = 16
        colors = {'B': 'darkorange', 'H': 'steelblue',
                  'T': 'seagreen', 'X': 'black', '-': 'grey'}
        lss = {True: '--', False: '-'}
        d = {True: 'syn', False: 'nonsyn'}
        labs = {'B': 'sheet', 'T': 'turn',
                'H': 'helix', 'X': 'unstructured'}
        for (syn, sec_str), datum in dav.unstack('time_binc').iterrows():
            datum = datum.unstack().T
            x = np.array(datum.index)
            y = np.array(datum['mean'])
            yerr = np.array(datum['std'])
            ax.errorbar(x, y,
                        yerr=yerr,
                        lw=3, ls=lss[syn], color=colors[sec_str],
                        label=d[syn]+', '+labs[sec_str])
        ax.set_xlim(0, 2600)
        ax.set_ylim(0, 0.015)
        ax.legend(loc='upper left', ncol=2, fontsize=fs-1)
        ax.set_xlabel('Days since EDI', fontsize=fs)
        ax.set_ylabel('Allele frequency', fontsize=fs)
        ax.xaxis.set_tick_params(labelsize=fs)
        ax.yaxis.set_tick_params(labelsize=fs)

        plt.tight_layout()
        plt.ion()
        plt.show()
