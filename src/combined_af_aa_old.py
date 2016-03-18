from __future__ import division, print_function
import sys
sys.path.append('/ebio/ag-neher/share/users/rneher/HIVEVO_access/')
import numpy as np
from itertools import izip
from hivevo.patients import Patient
from hivevo.sequence import alphaal
from hivevo.HIVreference import HIVreferenceAminoacid, HIVreference
from hivevo.af_tools import divergence
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, ks_2samp
from random import sample
import pandas as pd

fs=16
plt.ion()

nuc_muts = {
'A->C': 8.565560e-07,
'A->G': 5.033925e-06,
'A->T': 5.029735e-07,
'C->A': 3.062461e-06,
'C->G': 5.314500e-07,
'C->T': 1.120391e-05,
'G->A': 1.467470e-05,
'G->C': 1.987029e-07,
'G->T': 1.180130e-06,
'T->A': 1.708172e-06,
'T->C': 7.372737e-06,
'T->G': 1.899181e-06,
}


def patient_bootstrap(afs):
    '''
    resample allele frequencies of patients to produce pseudo replicates of equal size
    '''
    patients = afs.keys()
    tmp_sample = np.random.randint(len(patients), size=len(patients))
    return [afs[patients[ii]] for ii in tmp_sample]

def patient_partition(afs):
    '''
    take a set of allele frequencies for different patients and split
    it into two random partitions of equal size (like bootstraping but without resampling)
    '''
    patients = afs.keys()
    tmp_sample = sample(patients, len(patients)//2)
    remainder = set(patients).difference(tmp_sample)
    return [afs[pat] for pat in tmp_sample], [afs[pat] for pat in remainder]

def af_average(afs):
    '''
    average weighted allele frequency estimates.
    '''
    tmp_afs = np.sum(afs, axis=0)
    tmp_afs = tmp_afs/(np.sum(tmp_afs, axis=0)+1e-6)
    return tmp_afs

def aminoacid_mutation_rate(initial_codon, der, nuc_muts, doublehit=False):
    from Bio.Seq import CodonTable
    CT = CodonTable.standard_dna_table.forward_table
    targets = [x for x,a in CT.iteritems() if a==der]
    print(CT[initial_codon], initial_codon, targets)
    mut_rate=0
    for codon in targets:
        nmuts = sum([a!=d for a,d in zip(initial_codon, codon)])
        mut_rate+=np.prod([nuc_muts[a+'->'+d] for a,d in
                    zip(initial_codon, codon) if a!=d])*((nmuts==1) or doublehit)
        print(mut_rate, [nuc_muts[a+'->'+d] for a,d in zip(initial_codon, codon) if a!=d])

    return mut_rate

drug_muts = {'PR':{'offset': 56 - 1,
                    'mutations':  [('L', 24, 'I'), ('V', 32, 'I'), ('M', 46, 'IL'), ('I', 47, 'VA'),
                        ('G', 48, 'VM'), ('I', 50, 'LV'), ('I', 54, 'VTAM'), ('L', 76, 'V'),
                        ('V', 82, 'ATSF'), ('I', 84, 'V'), ('N', 88, 'S'), ('L', 90, 'M')]},
             'RT_NRTI':{'offset':56 + 99 - 1,
                    'mutations': [('M', 41, 'L'),('K', 65, 'R'),('K', 70, 'ER'),('L', 74, 'VI'),
                                ('Y', 115, 'F'),  ('M', 184,'VI'), ('L', 210,'W'), ('T', 215,'YF'), ('K', 219,'QE')]
                   },
             'RT_NNRTI':{'offset':56 + 99 - 1,
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

if __name__=="__main__":

    patients = ['p1', 'p2','p3','p4', 'p5','p6','p8', 'p9','p10', 'p11'] # all subtypes
    #patients = ['p2','p3','p5', 'p8', 'p9','p10', 'p11'] # subtype B only
    #regions = ['PR', 'RT'] #, 'p17', 'p24', 'PR', 'IN']
    regions = ["gag", "pol", "nef"]
    #regions = ['gp120', 'gp41']
    subtype='any'

    hla_assoc = pd.read_csv("data/Carlson_el_al_2012_HLAassoc.csv")
    #hla_assoc = pd.read_csv("data/Brumme_et_al_HLAassoc.csv")

    cov_min=100
    combined_af={}
    combined_af_by_pat={}
    initial_codons_by_pat={}
    combined_phenos={}
    aa_ref = 'NL4-3'
    for region in regions:
        combined_af_by_pat[region] = {}
        initial_codons_by_pat[region]={}
        nl43 = HIVreferenceAminoacid(region, refname=aa_ref, subtype = 'B')
        #hxb2 = HIVreference(refname='HXB2', subtype = 'B')
        good_pos_in_reference = nl43.get_ungapped(threshold = 0.05)
        combined_phenos[region] = np.zeros((len(nl43.entropy), 4))
        for pi, pcode in enumerate(patients):
            combined_af_by_pat[region][pcode] = np.zeros(nl43.af.shape)
            try:
                p = Patient.load(pcode)
            except:
                print("Can't load patient", pcode)
            else:
                print(pcode, p.Subtype)
                try:
                    aft = p.get_allele_frequency_trajectories(region, cov_min=cov_min, type='aa', error_rate=1e-3)
                except:
                    print("Can't load allele freqs of patient",pcode)
                    continue
                # get patient to subtype map and subset entropy vectors, convert to bits
                patient_to_subtype = p.map_to_external_reference_aminoacids(region, refname = aa_ref)
                init_nuc_sec = "".join(p.get_initial_sequence(region))
                initial_codons_by_pat[region][pcode] = {ci:init_nuc_sec[ii*3:(ii+1)*3] for ci, ii in patient_to_subtype}
                consensus = nl43.get_consensus_indices_in_patient_region(patient_to_subtype)
                ancestral = p.get_initial_indices(region, type='aa')[patient_to_subtype[:,1]]
                rare = ((aft[:,:21,:]**2).sum(axis=1).min(axis=0)>0.25)[patient_to_subtype[:,1]]
                final = aft[-1].argmax(axis=0)[patient_to_subtype[:,1]]
                do=[]
                acc=[]
                struct=[]
                for pos in p.annotation[region]:
                    if pos%3==1:
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
                acc = np.array(map(lambda x:0.0 if x is None else x, acc))
                struct = np.array(map(lambda x:0.0 if x is None else x, struct))
                combined_phenos[region][patient_to_subtype[:,0]-patient_to_subtype[0][0],1]+= do[patient_to_subtype[:,1]]
                combined_phenos[region][patient_to_subtype[:,0]-patient_to_subtype[0][0],2]+= acc[patient_to_subtype[:,1]]
                combined_phenos[region][patient_to_subtype[:,0]-patient_to_subtype[0][0],3]+= struct[patient_to_subtype[:,1]]


                for af, ysi, depth in izip(aft, p.ysi, p.n_templates_dilutions):
                    if ysi<1:
                        continue
                    pat_af = af[:,patient_to_subtype[:,1]]
                    patient_consensus = pat_af.argmax(axis=0)
                    ind = rare&(patient_consensus==consensus)&(ancestral==consensus)&(final==consensus) #(patient_consensus!=consensus) #&(ancestral==consensus)
                    combined_af_by_pat[region][pcode][:,patient_to_subtype[ind,0]-patient_to_subtype[0][0]] \
                                += min(300,depth)*pat_af[:-1,ind]

    combined_entropy={}
    minor_af={}
    combined_entropy_bs={}
    minor_af_bs={}
    combined_entropy_part={}
    minor_af_part={}
    nbs = 100
    for region in regions:
        combined_af[region] = af_average(combined_af_by_pat[region].values())
        combined_entropy[region] = (-np.log(combined_af[region]+1e-10)*combined_af[region]).sum(axis=0)
        minor_af[region] = (combined_af[region][:21,:].sum(axis=0) - combined_af[region].max(axis=0))/(1e-6+combined_af[region][:21,:].sum(axis=0))
        minor_af_bs[region]=[]
        combined_entropy_bs[region]=[]
        for ii in xrange(nbs):
            tmp_af = af_average(patient_bootstrap(combined_af_by_pat[region]))
            combined_entropy_bs[region].append((-np.log(tmp_af+1e-10)*tmp_af).sum(axis=0))
            minor_af_bs[region].append((tmp_af[:21,:].sum(axis=0) - tmp_af.max(axis=0))/(tmp_af[:21,:].sum(axis=0)+1e-6))
        minor_af_part[region]=[]
        combined_entropy_part[region]=[]
        for ii in xrange(nbs):
            for a in patient_partition(combined_af_by_pat[region]):
                tmp_af = af_average(a)
                combined_entropy_part[region].append((-np.log(tmp_af+1e-10)*tmp_af).sum(axis=0))
                minor_af_part[region].append((tmp_af[:21,:].sum(axis=0) - tmp_af.max(axis=0))/(tmp_af[:21,:].sum(axis=0)+1e-6))


        if region=='pol':
            for prot in drug_muts:
                drug_afs = {}
                drug_mut_rates = {}
                offset = drug_muts[prot]['offset']
                all_muts =  drug_muts[prot]['mutations']
                for cons_aa, pos, muts in all_muts:
                    tmp = []
                    tmp_muts = []
                    for pcode in combined_af_by_pat[region]:
                        drug_af = 0
                        mut_rate = 0
                        af_vec = combined_af_by_pat[region][pcode][:,pos+offset]
                        tot=af_vec.sum()
                        init_codon = initial_codons_by_pat[region][pcode][pos+offset]
                        if af_vec.argmax()!=alphaal.index(cons_aa):
                            print('Doesn')
                        if tot:
                            for aa in muts:
                                print(prot, pcode, pos, cons_aa)
                                mut_rate += aminoacid_mutation_rate(init_codon, aa, nuc_muts, doublehit=True)
                                drug_af+=af_vec[alphaal.index(aa)]/tot
                            tmp.append(drug_af)
                            tmp_muts.append(mut_rate)
                        else:
                            tmp.append(np.nan)
                            tmp_muts.append(np.nan)

                    drug_afs[(cons_aa,pos,muts)] = np.array(tmp)
                    drug_mut_rates[(cons_aa,pos,muts)] = np.array(tmp_muts)
                plt.figure()
                plt.title(prot)
                drug_afs_items = sorted(drug_afs.items(), key=lambda x:x[0][1])
                sns.stripplot(data=pd.DataFrame(np.array([x[1] for x in drug_afs_items]).T,
                              columns = ["".join(map(str,x[0])) for x in drug_afs_items]), jitter=True)
                plt.yscale('log')
                plt.ylim([3e-6, 1e-1])
                plt.ylabel('minor variant frequency')
                #plt.xticks(np.arange(len(all_muts)), ["".join(map(str, x)) for x in all_muts], rotation=60)
                plt.tight_layout()
                plt.savefig('figures/'+prot+'_minor_variant_frequency.pdf')

                plt.figure()
                plt.title(prot)
                drug_mut_rates_items = sorted(drug_mut_rates.items(), key=lambda x:x[0][1])
                sns.stripplot(data=pd.DataFrame(np.array([x[1] for x in drug_mut_rates_items]).T,
                              columns = ["".join(map(str,x[0])) for x in drug_mut_rates_items]), jitter=True)
                plt.yscale('log')
                plt.ylim([3e-8, 1e-4])
                plt.ylabel('mutation rate')
                #plt.xticks(np.arange(len(all_muts)), ["".join(map(str, x)) for x in all_muts], rotation=60)
                plt.tight_layout()
                plt.savefig('figures/'+prot+'_mutation_rate.pdf')


                aa_sel_coeff = sorted([(mut, np.mean(drug_mut_rates[mut])/np.mean(drug_afs[mut])) for mut in drug_afs], key=lambda x:x[0][1])
                plt.figure()
                plt.title(prot)
                plt.plot([min(1,a[1]) for a in aa_sel_coeff], 'o')
                plt.xticks(np.arange(len(aa_sel_coeff)), ["".join(map(str,a[0])) for a in aa_sel_coeff])
                plt.yscale('log')
                plt.ylabel('estimated fitness cost')
                #plt.xticks(np.arange(len(all_muts)), ["".join(map(str, x)) for x in all_muts], rotation=60)
                plt.tight_layout()
                plt.savefig('figures/'+prot+'_cost.pdf')
                print(aa_sel_coeff)
            #plt.boxplot(RT_afs)


    cols = sns.color_palette()
    for region in regions:
        nl43 = HIVreferenceAminoacid(region, refname=aa_ref, subtype = 'B')
        xsS = nl43.entropy
#        if region=='pol':
#            positions = []
#            for pr, offset  in [('PR', -1), ('RT', 98), ('INT', 658)]:
#                subset = (hla_assoc.loc[:,"Protein"]==pr)*(hla_assoc.loc[:,"q-value"]<0.1)
#                positions.extend(np.unique(hla_assoc.loc[subset, "Codon"])+offset+56)
#            hla_assoc_pos = np.in1d(np.arange(len(xsS)), np.unique(positions))
#        else:
#            subset = (hla_assoc.loc[:,"Protein"]==region.upper())*(hla_assoc.loc[:,"q-value"]<0.1)
#            hla_assoc_pos = np.in1d(np.arange(len(xsS)), np.unique(hla_assoc.loc[subset, "Codon"])-1)

        subset = (hla_assoc.loc[:,"Protein"]==region)*(hla_assoc.loc[:,"QValue"]<0.1)
        subset = (hla_assoc.loc[:,"Protein"]==region)*(np.abs(hla_assoc.loc[:,'PhyloD OR'])>10.0)
        if region=="pol":
            hla_assoc_pos = np.in1d(np.arange(len(xsS)), np.unique(hla_assoc.loc[subset, "Position"])-1+56)
        else:
            hla_assoc_pos = np.in1d(np.arange(len(xsS)), np.unique(hla_assoc.loc[subset, "Position"])-1)

        ind = xsS>=0.000
        print(region)
        print(np.corrcoef(combined_entropy[region][ind], xsS[ind]))
        print(spearmanr(combined_entropy[region][ind], xsS[ind]))
        ind = hla_assoc_pos
        print(np.corrcoef(combined_entropy[region][ind], xsS[ind]))
        print(spearmanr(combined_entropy[region][ind], xsS[ind]))
        ind = ~hla_assoc_pos
        print(np.corrcoef(combined_entropy[region][ind], xsS[ind]))
        print(spearmanr(combined_entropy[region][ind], xsS[ind]))

        plt.figure()
        #plt.title(region+' entropy scatter')
        npoints=25
        for tmp_ind, l, c in [(~hla_assoc_pos,"not", cols[1]),(hla_assoc_pos,"", cols[0])]:
            plt.scatter(combined_entropy[region][tmp_ind]+.00003, xsS[tmp_ind]+.005, c=c, label=l+" HLA associated", s=40)
            A = np.array(sorted(zip(combined_entropy[region][tmp_ind]+.00003, xsS[tmp_ind]+0.005), key=lambda x:x[0]))
            plt.plot(np.exp(np.convolve(np.log(A[:,0]+.00003), 1.0*np.ones(npoints)/npoints, mode='valid')),
                        np.exp(np.convolve(np.log(A[:,1]+.005), 1.0*np.ones(npoints)/npoints, mode='valid')), c=c, label=l+" HLA associated", lw=3)
        A = np.array((combined_entropy[region]+0.00003, xsS+0.005)).T
        prots = sorted([(k,offsets[k]) for k in protective_positions[region]], key=lambda x:x[1])
        prot_pos = np.array([x[1] for x in prots]+[1000000])

#        for ni, (intra, cross) in enumerate(A):
#            if intra<0.003 and cross>0.1:
#                prot = prots[np.argmax(prot_pos>ni)-1]
#                plt.annotate(prot[0]+':' +str(ni-prot[1]), (intra, cross), (intra*1.05, cross*1.05), color='g')

        ppos = []
        for feat, positions in protective_positions[region].iteritems():
            for pos in positions:
                intra, cross = A[pos+offsets[feat],:]
                ppos.append(pos+offsets[feat])
                plt.annotate(feat+':' +str(pos), (intra, cross), (intra*1.05, cross*1.05), color='r')
        ppos = np.in1d(np.arange(len(xsS)), np.unique(ppos))

        plt.ylabel('cross-sectional entropy', fontsize=fs)
        plt.xlabel('pooled within patient entropy', fontsize=fs)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylim([0.003, 2])
        plt.xlim([0.00001, .3])
        plt.tick_params(labelsize=fs*0.8)
        plt.tight_layout()
        plt.plot([1e-4,1], [1e-1,1])
        plt.savefig('figures/'+region+'_aa_entropy_scatter.pdf')

        diverse = (np.log(xsS+.00003) - 0.25*np.log(combined_entropy[region]+.00003))>0
        var_hla = combined_entropy[region][diverse&(ppos|hla_assoc_pos)]+.00003
        var_nonhla = combined_entropy[region][diverse&(~(ppos|hla_assoc_pos))]+.00003
        print("KS test region:", ks_2samp(var_hla, var_nonhla), len(var_hla), len(var_nonhla))
        plt.figure()
        plt.title(region)
        plt.plot(sorted(var_hla), np.linspace(0,1,len(var_hla)))
        plt.plot(sorted(var_nonhla), np.linspace(0,1,len(var_nonhla)))
        plt.xscale('log')

        plt.figure("minor_freq")
        plt.plot(sorted(minor_af[region]+0.00001), np.linspace(0,1,len(minor_af[region])),
                    label=region+' n='+str(len(minor_af[region])))
        plt.xscale('log')
        plt.yscale('linear')
        plt.legend(loc=2)
        plt.xlabel('minor frequency')
        plt.ylabel('P(nu<X)')
        plt.savefig('figures/aa_minor_af.pdf')

        plt.figure('section_coeff')
        plt.plot(sorted(4e-5/(minor_af[region]+0.0001)), np.linspace(1,0,len(minor_af[region])),
                label=region + ' n='+str(len(minor_af[region])))
        plt.xscale('log')
        plt.yscale('linear')
        plt.legend(loc=2)
        plt.xlabel('selection coeff')
        plt.ylabel('P(s<X)')

        plt.figure('section_coeff')
        sel_coeffs = [4e-5/(minor_af[region][pos+56+99-1]+0.0001) for pos in [103, 184, 190, 210, 215, 219]]
        plt.hist(4e-5/(minor_af[region]+0.0001), bins = np.logspace(-4,-0.5,31))
        plt.xscale('log')
        plt.yscale('linear')
        plt.legend(loc=2)
        plt.xlabel('selection coefficients')
        plt.ylabel('P(s<X)')

        plt.savefig('figures/aa_sel_coeff.pdf')

        plt.figure()
        minor_af_array=np.array(minor_af_bs[region])
        plt.title(region+' min/max frequency of boot strap replicates')
        minor_af_extrema = np.vstack((minor_af_array.min(axis=0), np.median(minor_af_array, axis=0), minor_af_array.max(axis=0)))
        sorted_by_median = np.array(sorted(minor_af_extrema.T, key=lambda x:x[1]))
        plt.plot(sorted_by_median[:,0], label='minimum')
        plt.plot(sorted_by_median[:,1], label='median')
        plt.plot(sorted_by_median[:,2], label='maximum')
        plt.yscale('log')
        plt.ylabel('frequency')
        plt.xlabel('positions sorted by median')
        plt.legend(loc=2)
        plt.savefig('figures/'+region + '_frequency_noise.pdf')


        nl43 = HIVreferenceAminoacid(region, refname='NL4-3', subtype = 'B')
        xsS = nl43.entropy
        for col, name in [(1,'disorder'), (2, 'accessibility'), (3, 'structural constraint')]:
            plt.figure()
            tstr = region + ' rho = '
            for tmp_ind, l, c in [(hla_assoc_pos,"", 'g'),  (~hla_assoc_pos,"not", 'r')]:
                ind = (combined_phenos[region][:,col]>0)&(tmp_ind)
                vals2 = (combined_phenos[region][ind,col]+0.01)
                vals1 = np.log10(xsS[ind]+1e-4)
                vals1 = np.log10(minor_af[region][ind]+1e-4)
                tstr += l+" "+str(np.round(spearmanr(vals1,vals2)[0], 2))
                plt.scatter(vals1, vals2, c=c)
                A = np.array(sorted(zip(vals1, vals2), key=lambda x:x[0]))
                plt.plot(np.convolve(A[:,0], 1.0*np.ones(npoints)/npoints, mode='valid'),
                         np.convolve(A[:,1], 1.0*np.ones(npoints)/npoints, mode='valid'), lw=3, c=c)

            plt.ylabel(name+' [log10]')
            plt.xlabel('minor af [log10]')
            plt.title(tstr)



from Bio.Seq import CodonTable
CT = CodonTable.standard_dna_table.forward_table
aa_mutation_rates = {}
for aa1 in alphaal:
    for aa2 in alphaal:
        if aa1!=aa2:
            tmp = []
            for codon in CT:
                if CT[codon]==aa1:
                    tmp.append(aminoacid_mutation_rate(codon, aa2, nuc_muts))
            aa_mutation_rates[aa1+'->'+aa2] = tmp
