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


# Functions
def parse_secondary_structure(filename):
    pass


# Script
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Secondary structure')
    parser.add_argument('--protein',
                        choices=['gagpol'],
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

    # Load annotations
    t = pd.read_csv(fn,
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

    # Apply annotations
    from Bio.SeqFeature import SeqFeature, FeatureLocation
    for _, datum in t.iterrows():
        anno = SeqFeature(FeatureLocation(datum['start'],datum['end']), strand=+1)
        anno.type = datum['feature']
        seq.features.append(anno)
