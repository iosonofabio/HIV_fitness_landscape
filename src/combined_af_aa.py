from __future__ import division, print_function
import sys, argparse, cPickle, os, gzip
sys.path.append('/ebio/ag-neher/share/users/rneher/HIVEVO_access/')
import numpy as np
from itertools import izip
from hivevo.patients import Patient
from hivevo.sequence import alphaal
from hivevo.HIVreference import HIVreferenceAminoacid, HIVreference
from hivevo.af_tools import divergence
from matplotlib import pyplot as plt
from collections import defaultdict
import seaborn as sns
from scipy.stats import spearmanr, scoreatpercentile, pearsonr
from random import sample
import pandas as pd
from combined_af import process_average_allele_frequencies

fs=16
ls = {'gag':'-', 'pol':'--', 'nef':'-.'}
cols = sns.color_palette()
plt.ion()


def aminoacid_mutation_rate(initial_codon, der, nuc_muts, doublehit=False):
    from Bio.Seq import CodonTable
    CT = CodonTable.standard_dna_table.forward_table
    targets = [x for x,a in CT.iteritems() if a==der]
    #print(CT[initial_codon], initial_codon, targets)
    mut_rate=0
    for codon in targets:
        nmuts = sum([a!=d for a,d in zip(initial_codon, codon)])
        mut_rate+=np.prod([nuc_muts[a+'->'+d] for a,d in
                    zip(initial_codon, codon) if a!=d])*((nmuts==1) or doublehit)
        #print(mut_rate, [nuc_muts[a+'->'+d] for a,d in zip(initial_codon, codon) if a!=d])

    return mut_rate

def calc_amino_acid_mutation_rates():
    from Bio.Seq import CodonTable

    with open('data/mutation_rate.pickle') as mutfile:
        nuc_mutation_rates = cPickle.load(mutfile)['mu']

    CT = CodonTable.standard_dna_table.forward_table
    aa_mutation_rates = defaultdict(float)
    total_mutation_rates = defaultdict(float)
    for codon in CT:
        aa1 = CT[codon]
        for aa2 in alphaal:
            if aa1!=aa2:
                aa_mutation_rates[(codon,aa2)] += aminoacid_mutation_rate(codon, aa2, nuc_mutation_rates)

    for codon,aa1 in CT.iteritems():
        for pos in range(3):
            for nuc in 'ACTG':
                new_codon= codon[:pos]+nuc+codon[(pos+1):]
                if new_codon in CT:
                    if aa1!=CT[new_codon]:
                        total_mutation_rates[codon]+=nuc_mutation_rates[codon[pos]+'->'+nuc]
    return aa_mutation_rates, total_mutation_rates


drug_muts = {'PI':{'offset': 56 - 1,
                    'mutations':  [('L', 24, 'I'), ('V', 32, 'I'), ('M', 46, 'IL'), ('I', 47, 'VA'),
                        ('G', 48, 'VM'), ('I', 50, 'LV'), ('I', 54, 'VTAM'), ('L', 76, 'V'),
                        ('V', 82, 'ATSF'), ('I', 84, 'V'), ('N', 88, 'S'), ('L', 90, 'M')]},
             'NRTI':{'offset':56 + 99 - 1,
                    'mutations': [('M', 41, 'L'),('K', 65, 'R'),('K', 70, 'ER'),('L', 74, 'VI'),
                                ('Y', 115, 'F'),  ('M', 184,'VI'), ('L', 210,'W'), ('T', 215,'YF'), ('K', 219,'QE')]
                   },
             'NNRTI':{'offset':56 + 99 - 1,
                    'mutations': [('L', 100, 'I'),('K', 101, 'PEH'), ('K', 103,'N'),
                                ('V', 106, 'AM'),('E', 138, 'K'),('V', 179, 'DEF'), ('Y', 181, 'CIV'),
                                ('Y', 188, 'LCH'),('G',190,'ASEQ'), ('F', 227,'LC'), ('M', 230,'L')]
                   }
            }
protective_positions = {
    'gag':{'gag':[12, 26, 28,79,147, 242, 248, 264, 268, 340, 357, 397, 403, 437]},
    'nef':{'nef':[71,81, 83, 85, 92, 94, 102,105,116,120,126,133,135]},
    'pol':{
    'PR':[35,93],
    'RT':[135, 245,277,369,395],
    'INT':[11,32,119,122,124],}
}
offsets = {
    'gag':-1,
    'PR':55,
    'nef':-1,
    'RT':55+99,
    'INT':55+99+440+120,
}


def collect_weighted_aa_afs(region, patients, reference, cov_min=1000, max_div=0.05):
    '''
    produce weighted averages of allele frequencies for all late samples in each patients
    restrict to sites that don't sweep and have limited diversity as specified by max_div
    '''
    combined_af_by_pat = {}
    initial_codons_by_pat={}
    combined_phenos = {'disorder':np.zeros(len(reference.entropy)),
                        'accessibility':np.zeros(len(reference.entropy)),
                        'structural':np.zeros(len(reference.entropy))}

    good_pos_in_reference = reference.get_ungapped(threshold = 0.05)
    for pi, p in enumerate(patients):
        pcode = p.name
        combined_af_by_pat[pcode] = np.zeros(reference.af.shape)
        aft = p.get_allele_frequency_trajectories(region, cov_min=cov_min, type='aa', error_rate=2e-3)

        # get patient to subtype map and initial aa and nuc sequence
        patient_to_subtype = p.map_to_external_reference_aminoacids(region, refname = reference.refname)
        init_nuc_sec = "".join(p.get_initial_sequence(region))
        consensus = reference.get_consensus_indices_in_patient_region(patient_to_subtype)
        ref_ungapped = good_pos_in_reference[patient_to_subtype[:,0]]

        # remember the codon at each reference position to be able to calculate mutation rates later
        initial_codons_by_pat[pcode] = {ci:init_nuc_sec[ii*3:(ii+1)*3] for ci, ii in patient_to_subtype}

        ancestral = p.get_initial_indices(region, type='aa')[patient_to_subtype[:,1]]
        rare = ((aft[:,:21,:]**2).sum(axis=1).min(axis=0)>max_div)[patient_to_subtype[:,1]]
        final = aft[-1].argmax(axis=0)[patient_to_subtype[:,1]]

        do=[]
        acc=[]
        struct=[]
        for pos in p.annotation[region]:
            if pos%3==1: # extract phenotypes for each
                try:
                    do.append(np.mean(p.pos_to_feature[pos]['disorder'].values()))
                except:
                    do.append(None)
                try:
                    struct.append(np.mean(p.pos_to_feature[pos]['structural'].values()))
                except:
                    struct.append(None)
                try:
                    acc.append(np.mean(p.pos_to_feature[pos]['accessibility'].values()))
                except:
                    acc.append(None)
        do = np.array(map(lambda x:0.0 if x is None else x, do))
        combined_phenos['disorder'][patient_to_subtype[:,0]]+= do[patient_to_subtype[:,1]]

        acc = np.array(map(lambda x:0.0 if x is None else x, acc))
        combined_phenos['accessibility'][patient_to_subtype[:,0]]+= acc[patient_to_subtype[:,1]]

        struct = np.array(map(lambda x:0.0 if x is None else x, struct))
        combined_phenos['structural'][patient_to_subtype[:,0]]+= struct[patient_to_subtype[:,1]]

        for af, ysi, depth in izip(aft, p.ysi, p.n_templates_dilutions):
            if ysi<1:
                continue
            pat_af = af[:,patient_to_subtype[:,1]]
            patient_consensus = pat_af.argmax(axis=0)
            ind = ref_ungapped&rare&(patient_consensus==consensus)&(ancestral==consensus)&(final==consensus)
            w = depth/(1.0+depth/300.0)
            combined_af_by_pat[pcode][:,patient_to_subtype[ind,0]] \
                        += w*pat_af[:-1,ind]
    for pheno in combined_phenos:
        combined_phenos[pheno]/=len(patients)
    return combined_af_by_pat, initial_codons_by_pat, combined_phenos


def collect_data(patient_codes, regions):
    cov_min=1000
    combined_af_by_pat={}
    initial_codons_by_pat={}
    combined_phenos={}
    aa_ref = 'NL4-3'
    patients = []

    for pcode in patient_codes:
        try:
            p = Patient.load(pcode)
            patients.append(p)
        except:
            print("Can't load patient", pcode)

    for region in regions:
        reference = HIVreferenceAminoacid(region, refname=aa_ref, subtype = 'B')
        combined_af_by_pat[region], initial_codons_by_pat[region], combined_phenos[region] =\
            collect_weighted_aa_afs(region, patients, reference, cov_min=cov_min)

    return {'af_by_pat':combined_af_by_pat, 'init_codon': initial_codons_by_pat, 'pheno':combined_phenos}

def get_associations(regions, aa_ref='NL4-3'):
    hla_assoc = pd.read_csv("data/Carlson_el_al_2012_HLAassoc.csv")
    #hla_assoc = pd.read_csv("data/Brumme_et_al_HLAassoc.csv")
    associations = {}
    for region in regions:
        reference = HIVreferenceAminoacid(region, refname=aa_ref, subtype = 'B')
        L = len(reference.entropy)
        associations[region]={}
        subset = (hla_assoc.loc[:,"Protein"]==region)*(hla_assoc.loc[:,"QValue"]<0.1)
        if region=="pol":
            hla_assoc_pos = np.in1d(np.arange(L), np.unique(hla_assoc.loc[subset, "Position"])-1+56)
        else:
            hla_assoc_pos = np.in1d(np.arange(L), np.unique(hla_assoc.loc[subset, "Position"])-1)
        associations[region]['HLA'] = hla_assoc_pos
        ppos = []
        for feat, positions in protective_positions[region].iteritems():
            for pos in positions:
                ppos.append(pos+offsets[feat])
        associations[region]['protective'] = np.in1d(np.arange(L), np.unique(ppos))
    return associations

def entropy_scatter(region, within_entropy, associations, reference, fname = None, annotate_protective=False):
    '''
    scatter plot of cross-sectional entropy vs entropy of averaged
    intrapatient frequencies amino acid frequencies
    '''
    xsS = reference.entropy
    ind = xsS>=0.000
    print(region)
    print("Pearson:", pearsonr(within_entropy[region][ind], xsS[ind]))
    rho, pval = spearmanr(within_entropy[region][ind], xsS[ind])
    print("Spearman:", rho, pval)

    plt.figure(figsize = (7,6))
    assoc_ind = associations[region]['HLA']|associations[region]['protective']
    for ni, tmp_imd, label_str in ((2, ~assoc_ind, 'other'), (0, assoc_ind, 'HLA/protective')):
        ind = (xsS>=0.000)&tmp_imd
        plt.scatter(within_entropy[region][ind]+.00003, xsS[ind]+.005, c=cols[ni], label=label_str, s=30)
    if annotate_protective:
        A = np.array((combined_entropy[region]+0.00003, xsS+0.005)).T
        for feat, positions in protective_positions[region].iteritems():
            for pos in positions:
                intra, cross = A[pos+offsets[feat],:]
                plt.annotate(feat+':' +str(pos), (intra, cross), (intra*1.05, cross*1.05), color='r')

    plt.ylabel('cross-sectional entropy', fontsize=fs)
    plt.xlabel('pooled within patient entropy', fontsize=fs)
    plt.text(0.00002, 3, r"Combined Spearman's $\rho="+str(round(rho,2))+"$", fontsize=fs)
    plt.legend(loc=4, fontsize=fs*0.8)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim([0.001, 4])
    plt.xlim([0.00001, .3])
    plt.tick_params(labelsize=fs*0.8)
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)

def phenotype_scatter(region, within_entropy, phenotype, phenotype_name, fname = None):
    ind = phenotype!=0
    print(region, phenotype_name)
    print("Pearson:", pearsonr(within_entropy[region][ind], phenotype[ind]))
    rho, pval = spearmanr(within_entropy[region][ind], phenotype[ind])
    print("Spearman:", rho, pval)

    plt.figure(figsize = (7,6))
    plt.title("Spearman's rho: "+str(np.round(rho,2)))
    plt.scatter(within_entropy[region][ind]+.00003, phenotype[ind], s=30)

    plt.xlabel('pooled within patient entropy', fontsize=fs)
    plt.ylabel(phenotype_name, fontsize=fs)
    plt.xscale('log')
    plt.xlim([0.00001, .3])
    plt.tick_params(labelsize=fs*0.8)
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)


def selection_coefficient_mutation(region, data, aa_mutation_rates, pos, target_aa):
    target_ii = alphaal.index(target_aa)
    codons = data['init_codon'][region]
    minor_af_by_pat = {pat: x[target_ii,pos].sum(axis=0)/x[:20,pos].sum(axis=0)
                        for pat, x in data['af_by_pat'][region].iteritems() if pos in codons[pat]}

    print(pos, target_aa,[codons[pat][pos] for pat in codons], [aa_mutation_rates[(codons[pat][pos],target_aa)] for pat in codons])
    nu_over_mu = [(1e-5+nu)/aa_mutation_rates[(codons[pat][pos],target_aa)] for pat, nu in minor_af_by_pat.iteritems()]
    return 1.0/np.mean(nu_over_mu)


def selection_coefficients_per_site(region,data, total_nonsyn_mutation_rates):
    nu_over_mu = []
    codons = data['init_codon'][region]
    minor_af_by_pat = {pat: (x[:20,:].sum(axis=0) - x[:20,:].max(axis=0))/x[:20,:].sum(axis=0)
                        for pat, x in data['af_by_pat'][region].iteritems()}

    for pat in minor_af_by_pat:
        tmp=[]
        for pos, nu in enumerate(minor_af_by_pat[pat]):
            if pos in codons[pat]:
                tmp.append(nu/total_nonsyn_mutation_rates[codons[pat][pos]])
            else:
                tmp.append(np.nan)

        nu_over_mu.append(tmp)

    nu_over_mu = np.ma.array(nu_over_mu)
    nu_over_mu.mask = np.isnan(nu_over_mu)
    return 1.0/nu_over_mu.mean(axis=0)

def selection_coefficients_distribution(region, data, total_nonsyn_mutation_rates):
    selcoeff = selection_coefficients_per_site(region, data, total_nonsyn_mutation_rates)
    selcoeff[selcoeff<0.001]=0.001
    selcoeff[selcoeff>0.1]=0.1

    n=selcoeff.shape[0]
    plt.figure(figsize=(8,6))
    plt.hist(selcoeff, weights=np.ones(n, dtype=float)/n, bins=np.logspace(-3,-1,11))
    plt.xscale('log')


def compare_experiments(data, aa_mutation_rates):
    fc = pd.read_csv('data/fitness_costs.csv')
    coefficients = {}
    for ii, mut in fc.iterrows():
        region = mut['region']
        offset = offsets[mut['feature']]
        aa, pos = mut['mutation'][-1], int(mut['mutation'][1:-1])+offset
        coefficients[(mut['feature'], mut['mutation'])] = (mut['normalized'],
            selection_coefficient_mutation(region, data, aa_mutation_rates, pos, aa))

    return coefficients


def plot_drug_resistance_mutations(data, aa_mutation_rates):
    import matplotlib.patches as patches
    region='pol'
    pcodes = data['init_codon'][region].keys()
    fig = plt.figure()
    ax=plt.subplot(111)
    drug_afs_items = []
    mut_types = []
    drug_classes = ['PI', 'NRTI', 'NNRTI']
    for prot in drug_classes:
        drug_afs = {}
        drug_mut_rates = {}
        offset = drug_muts[prot]['offset']
        for cons_aa, pos, target_aa in drug_muts[prot]['mutations']:
            codons = {pat:data['init_codon'][region][pat][pos+offset] for pat in pcodes}
            mut_rates = {pat:np.sum([aa_mutation_rates[(codons[pat], aa)] for aa in target_aa])
                        for pat in pcodes}
            freqs = {pat:np.sum([data['af_by_pat'][region][pat][alphaal.index(aa), pos+offset]/data['af_by_pat'][region][pat][:20,pos+offset].sum()
                        for aa in target_aa]) for pat in pcodes}


            drug_afs[(cons_aa,pos,target_aa)] = freqs
            drug_mut_rates[(cons_aa,pos,target_aa)] = mut_rates

        print(prot, drug_mut_rates,drug_afs)
        drug_afs_items.extend(filter(lambda x:np.sum(filter(lambda y:~np.isnan(y), x[1].values()))>0, sorted(drug_afs.items(), key=lambda x:x[0][1])))
        mut_types.append(len(drug_afs_items))
        print('Monomorphic:', prot, [''.join(map(str,x[0])) for x in filter(lambda x:np.sum(filter(lambda y:~np.isnan(y), x[1].values())    )==0, sorted(drug_afs.items(), key=lambda x:x[0][1]))])

    plt.ylim([3e-5, 1e-1])
    for mi in mut_types[:-1]:
        plt.plot([mi-0.5,mi-0.5], plt.ylim(), c=(.3,.3,.3), lw=3, alpha=0.5)

    for ni, prot in enumerate(drug_classes):
        plt.text(0.5*(mut_types[ni] + (mut_types[ni-1] if ni else 0))-0.5, 0.07, prot, fontsize=16, ha='center')

    for mi in range(max(mut_types)):
        c = 0.5 + 0.2*(mi%2)
        ax.add_patch( patches.Rectangle(
                (mi-0.5, plt.ylim()[0]),  1.0, plt.ylim()[1], #(x,y), width, height
                color=(c,c,c), alpha=0.2
            )
        )

    #plt.xticks(np.arange(len(all_muts)), ["".join(map(str, x)) for x in all_muts], rotation=60)
    sns.stripplot(data=pd.DataFrame(np.array([x[1].values() for x in drug_afs_items]).T,
                  columns = ["".join(map(str,x[0])) for x in drug_afs_items]), jitter=True)
    plt.yscale('log')
    plt.xticks(rotation=30)
    plt.ylabel('minor variant frequency', fontsize=fs)
    plt.tick_params(labelsize=fs*0.8)
    plt.tight_layout()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='amino acid allele frequencies, saturation levels, and fitness costs')
    parser.add_argument('--regenerate', action='store_true',
                        help="regenerate data")
    args = parser.parse_args()

    fn = 'data/avg_aa_allele_frequency.pickle.gz'

    regions = ['gag', 'pol', 'nef']
    if not os.path.isfile(fn) or args.regenerate:
        #patient_codes = ['p1', 'p2','p3','p5','p6', 'p8', 'p9','p10', 'p11'] # all subtypes, no p4/7
        #patient_codes = ['p1', 'p2','p3','p4', 'p5','p6','p7', 'p8', 'p9','p10', 'p11'] # patients
        patient_codes = ['p2','p3','p4','p5', 'p8', 'p9','p10', 'p11'] # subtype B only
        data = collect_data(patient_codes, regions)
        with gzip.open(fn, 'w') as ofile:
            cPickle.dump(data, ofile)
    else:
        with gzip.open(fn) as ifile:
            data = cPickle.load(ifile)

    combined_af, combined_entropy, minor_af = process_average_allele_frequencies(data, regions, nbootstraps=0, nstates=20)

    associations = get_associations(regions)
    aa_mutation_rates, total_nonsyn_mutation_rates = calc_amino_acid_mutation_rates()

    aa_ref='NL4-3'

#    for region in regions:
#        selection_coefficients_distribution(region, data, total_nonsyn_mutation_rates)
#
#        reference = HIVreferenceAminoacid(region, refname=aa_ref, subtype = 'B')
#        entropy_scatter(region, combined_entropy, associations, reference,'figures/'+region+'_aa_entropy_scatter.pdf', annotate_protective=True)
#
#        for phenotype, vals in data['pheno'][region].iteritems():
#            print(phenotype)
#            phenotype_scatter(region, combined_entropy, vals, phenotype)
#

plot_drug_resistance_mutations(data, aa_mutation_rates)


