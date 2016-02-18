# vim: fdm=indent
'''
author:     Fabio Zanini
date:       19/01/16
content:    Test for the secondary structure file, which is in a custom TSV
            format from uniprot (unbelievable).
'''
# Modules
import os
import sys
import argparse
import numpy as np
import pandas as pd
from Bio import SeqIO

from hivevo.HIVreference import HIVreference
from hivevo.patients import Patient

# Functions
def load_secondary_structure_patient(pcode):
    '''Load the genomewide array with the protein secondary structures'''
    fn = 'data/secondary_uniprot/patient_'+pcode+'.pickle'
    return pd.read_pickle(fn)


def parse_secondary_structure(protein):
    '''Parse protein secondary structure data'''
    filename = 'data/secondary_uniprot/'+protein+'.tsv'
    t = pd.read_csv(filename,
                    sep='\t',
                    usecols=[0, 1, 2],
                   )
    t.rename(columns={u'# Feature key': 'feature',
                      u'Position(s)': 'location',
                      u'Length': 'length'},
             inplace=True)
    t['start'] = t['location'].apply(lambda x: int(x.split(' ')[0]))
    t['end'] = t['location'].apply(lambda x: int(x.split(' ')[-1]) + 1)
    t.drop(['length', 'location'], inplace=True, axis=1)
    t.protein = args.protein
    return t


def annotate_dna_reference(ref, protein):
    '''Annotate DNA reference with the protein secondary structures'''
    from Bio.SeqFeature import SeqFeature, FeatureLocation

    annotation_table = parse_secondary_structure(protein)

    if protein == 'gagpol':
        start_protein = ref.annotation['gag'].location.nofuzzy_start
    elif protein == 'vpu':
        # Uniprot starts one aa downstream of us
        start_protein = ref.annotation[protein].location.nofuzzy_start + 3
    else:
        start_protein = ref.annotation[protein].location.nofuzzy_start

    features = []
    for _, datum in annotation_table.iterrows():
        start_dna = datum['start'] * 3 + start_protein
        end_dna = datum['end'] * 3 + start_protein

        # Notice ribosomal slippage site
        if protein == 'gagpol':
            if start_dna >= slippage_site:
                start_dna -= 1
            if end_dna >= slippage_site:
                end_dna -= 1

        anno = SeqFeature(FeatureLocation(start_dna, end_dna), strand=+1)
        anno.type = datum['feature']
        features.append(anno)

    return features


def double_check_reference(refprot, seq):
    refm = np.array(refprot)
    seqm = np.array(seq)

    L = 800
    for i in xrange(L // 100):
        s1 = refm[i * 100: (i+1) * 100]
        s2 = seqm[i * 100: (i+1) * 100]
        if len(s1) == len(s2):
            if len(s1) == 0:
                break
            print ''.join(s1)
            print ''.join(['x' if x else ' ' for x in (s1 != s2)])
            print ''.join(s2)
            print ''
        else:
            l = min(len(s1), len(s2))
            print ''.join(s1)
            print ''.join(['x' if x else ' ' for x in (s1[:l] != s2[:l])])
            print ''.join(s2)
            print ''
            break



# Script
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Secondary structure')
    parser.add_argument('--protein',
                        choices=['gagpol', 'env', 'nef', 'vpu'],
                        default='gagpol',
                        help="Protein to study")

    args = parser.parse_args()

    # Coordinates should be AA of HXB2, but we load the sequence to make
    # sure
    fn = 'data/secondary_uniprot/'+args.protein+'.tsv'
    fn_seq = 'data/secondary_uniprot/'+args.protein+'.fasta'

    # Get sequence first, then annotate (for now)
    # TODO: we will annotate the actual DNA sequence eventually
    seq = SeqIO.read(fn_seq, 'fasta')

    # Get the DNA sequence from our reference and compare (should be HXB2)
    refname = 'HXB2'
    ref = HIVreference(refname=refname, subtype='B', load_alignment=False)

    # Extract gagpol feature
    if args.protein == 'gagpol':
        from Bio.Seq import Seq
        start = ref.annotation['gag'].location.nofuzzy_start
        end = ref.annotation['pol'].location.nofuzzy_end
        slippage_site = 434
        refdna = (ref.seq[start: start + slippage_site * 3] +
                  ref.seq[start + slippage_site * 3 - 1: end])
        refprot = refdna.seq.translate()

    elif args.protein == 'nef':
        # it comes with the stop codon...
        refdna = ref.annotation[args.protein].extract(ref.seq)
        refprot = refdna.seq.translate()[:-1]

    elif args.protein == 'vpu':
        # it comes with a shift of one aa and the stop codon...
        refdna = ref.annotation[args.protein].extract(ref.seq)
        refprot = refdna.seq.translate()[1:-1]

    else:
        refdna = ref.annotation[args.protein].extract(ref.seq)
        refprot = refdna.seq.translate()

    double_check_reference(refprot, seq)

    #t = parse_secondary_structure(args.protein)

    features = annotate_dna_reference(ref, args.protein)

    #sys.exit()

    # Annotate patient sequences
    patients = ['p1', 'p2', 'p3','p5', 'p6', 'p8', 'p9', 'p11']
    for pi, pcode in enumerate(patients):
        print pcode
        p = Patient.load(pcode)

        if args.protein == 'gagpol':
            rois = ['gag', 'pol']
        else:
            rois = [args.protein]

        from collections import defaultdict

        # Add the annotations to the existing ones
        if False:
            pos_array = p.get_initial_sequence('genomewide')
            pos_array[:] = '-'
        else:
            fn = 'data/secondary_uniprot/patient_'+pcode+'.pickle'
            pos_array = np.array(pd.read_pickle(fn))

        # Set all positions in this protein as UNSTRUCTURED
        for roi in rois:
            m = p.map_to_external_reference(roi, refname='HXB2')[:, 1]
            pos_array[m] = 'X'

        for roi in rois:
            m = p.map_to_external_reference(roi, refname='HXB2')[:, :2]
            m = pd.Series(m[:, 1], index=m[:, 0])
            for fea in features:
                poss = [m.loc[pos] for pos in list(fea) if pos in m.index]
                pos_array[poss] = fea.type[0]

        pos_array = pd.Series(pos_array)
        pos_array.name = 'protein secondary structure'

        fn_out = 'data/secondary_uniprot/patient_'+pcode+'.pickle'
        pos_array.to_pickle(fn_out)
