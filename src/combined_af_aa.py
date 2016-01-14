from __future__ import division, print_function
import sys
sys.path.append('/ebio/ag-neher/share/users/rneher/HIVEVO_access/')
import numpy as np
from itertools import izip
from hivevo.patients import Patient
from hivevo.HIVreference import HIVreferenceAminoacid, HIVreference
from hivevo.af_tools import divergence
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from random import sample
import pandas as pd

plt.ion()
def patient_bootstrap(afs):
    patients = afs.keys()
    tmp_sample = np.random.randint(len(patients), size=len(patients))
    return [afs[patients[ii]] for ii in tmp_sample]

def patient_partition(afs):
    patients = afs.keys()
    tmp_sample = sample(patients, len(patients)//2)
    remainder = set(patients).difference(tmp_sample)
    return [afs[pat] for pat in tmp_sample], [afs[pat] for pat in remainder]


def af_average(afs):
    tmp_afs = np.sum(afs, axis=0)
    tmp_afs = tmp_afs/(np.sum(tmp_afs, axis=0)+1e-6)
    return tmp_afs


if __name__=="__main__":
    patients = ['p1', 'p2','p3','p4', 'p5','p6','p7', 'p8', 'p9','p10', 'p11'] # all subtypes
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
    combined_phenos={}
    aa_ref = 'NL4-3'
    for region in regions:
        combined_af_by_pat[region] = {}
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
        plt.title(region+' entropy scatter')
        npoints=25
        for tmp_ind, l, c in [(hla_assoc_pos,"", 'g'),  (~hla_assoc_pos,"not", 'r')]:
            plt.scatter(combined_entropy[region][tmp_ind]+.00001, xsS[tmp_ind]+.001, c=c, label=l+" HLA associated")
            A = np.array(sorted(zip(combined_entropy[region][tmp_ind], xsS[tmp_ind]), key=lambda x:x[0]))
            plt.plot(np.exp(np.convolve(np.log(A[:,0]+.00001), 1.0*np.ones(npoints)/npoints, mode='valid')),
                        np.exp(np.convolve(np.log(A[:,1]+.001), 1.0*np.ones(npoints)/npoints, mode='valid')), c=c, label=l+" HLA associated", lw=3)
        plt.ylabel('cross-sectional entropy')
        plt.xlabel('pooled within patient entropy')
        plt.yscale('log')
        plt.xscale('log')
        plt.ylim([0.001, 2])
        plt.xlim([0.0001, 2])
        plt.savefig('figures/'+region+'_aa_entropy_scatter.pdf')

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
        plt.plot(sorted(1e-5/(minor_af[region]+0.0001)), np.linspace(1,0,len(minor_af[region])),
                label=region + ' n='+str(len(minor_af[region])))
        plt.xscale('log')
        plt.yscale('linear')
        plt.legend(loc=2)
        plt.xlabel('selection coeff')
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
