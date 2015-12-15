from __future__ import division, print_function
import sys
sys.path.append('/ebio/ag-neher/share/users/rneher/HIVEVO_access/')
import numpy as np
from itertools import izip
from hivevo.patients import Patient
from hivevo.HIVreference import HIVreferenceAminoacid
from hivevo.af_tools import divergence
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from random import sample

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
    patients = ['p1', 'p2','p3','p5','p6', 'p8', 'p9','p10', 'p11'] # all subtypes
    #patients = ['p2','p3','p5', 'p8', 'p9','p10', 'p11'] # subtype B only
    regions = ['RT']
    subtype='any'

    cov_min=100
    combined_af={}
    combined_af_by_pat={}
    for region in regions:
        combined_af_by_pat[region] = {}
        nl43 = HIVreferenceAminoacid(region, refname='NL4-3', subtype = 'B')
        good_pos_in_reference = nl43.get_ungapped(threshold = 0.05)
        for pi, pcode in enumerate(patients):
            combined_af_by_pat[region][pcode] = np.zeros(nl43.af.shape)
            try:
                p = Patient.load(pcode)
            except:
                print("Can't load patient", pcode)
            else:
                print(pcode, p.Subtype)
                aft = p.get_allele_frequency_trajectories(region, cov_min=cov_min, type='aa', error_rate=1e-3)

                # get patient to subtype map and subset entropy vectors, convert to bits
                patient_to_subtype = p.map_to_external_reference_aminoacids(region, refname = 'NL4-3')
                consensus = nl43.get_consensus_indices_in_patient_region(patient_to_subtype)
                ancestral = p.get_initial_indices(region, type='aa')[patient_to_subtype[:,1]]
                rare = ((aft[:,:21,:]**2).sum(axis=1).min(axis=0)>0.5)[patient_to_subtype[:,1]]
                final = aft[-1].argmax(axis=0)[patient_to_subtype[:,1]]
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
        xsS = nl43.entropy
        ind = xsS>=0.000
        print(region)
        print(np.corrcoef(combined_entropy[region][ind], xsS[ind]))
        print(spearmanr(combined_entropy[region][ind], xsS[ind]))

        plt.figure()
        plt.title(region+' entropy scatter')
        plt.scatter(combined_entropy[region][ind]+.00001, xsS[ind]+.001)
        plt.ylabel('cross-sectional entropy')
        plt.xlabel('pooled within patient entropy')
        plt.yscale('log')
        plt.xscale('log')
        plt.ylim([0.001, 2])
        plt.xlim([0.0001, 2])
        plt.savefig('figures/'+region+'_entropy_scatter.pdf')

        plt.figure()
        plt.title(region+' minor freq')
        for ni in range(3):
            ind = (np.arange(len(minor_af[region]))%3)==ni
            plt.plot(sorted(minor_af[region][ind]+0.00001), np.linspace(0,1,ind.sum()),
                    label='Codon pos:'+str(ni)+' n='+str(np.sum(ind)))
        plt.xscale('log')
        plt.yscale('linear')
        plt.legend(loc=2)
        plt.xlabel('minor frequency')
        plt.ylabel('P(nu<X)')
        plt.savefig('figures/'+region+'_minor_af.pdf')

        plt.figure()
        plt.title(region+' selection coefficients')
        for ni in range(3):
            ind = (np.arange(len(minor_af[region]))%3)==ni
            plt.plot(sorted(1e-5/(minor_af[region][ind]+0.0001)), np.linspace(1,0,ind.sum()),
                    label='Codon pos:'+str(ni)+' n='+str(np.sum(ind)))
        plt.xscale('log')
        plt.yscale('linear')
        plt.legend(loc=2)
        plt.xlabel('selection coeff')
        plt.ylabel('P(s<X)')
        plt.savefig('figures/'+region+'_sel_coeff.pdf')


        ii=2
        plt.figure()
        ind = (~(np.isnan(minor_af_part[region][ii])|np.isnan(minor_af_part[region][ii+1])))&\
              ((minor_af_part[region][ii]>0) & (minor_af_part[region][ii+1]>0))
        plt.scatter(minor_af_part[region][ii][ind]+1e-6, minor_af_part[region][ii+1][ind]+1e-6) #, yscale='log', xscale='log')
        print(spearmanr(minor_af_part[region][ii][ind]+1e-6, minor_af_part[region][ii+1][ind]+1e-6))
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim([1e-6,1])
        plt.ylim([1e-6,1])

        ii=2
        plt.figure()
        plt.hist(np.log10(minor_af_part[region][ii][ind]+1e-6) - np.log10(minor_af_part[region][ii+1][ind]+1e-6), bins=30)
        print("Difference std:", np.std(np.log10(minor_af_part[region][ii][ind]+1e-6) - np.log10(minor_af_part[region][ii+1][ind]+1e-6)))
        print("Single std:", np.std(np.log10(minor_af_part[region][ii][ind]+1e-6)), np.std(np.log10(minor_af_part[region][ii+1][ind]+1e-6)))


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

