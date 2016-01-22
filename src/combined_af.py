from __future__ import division, print_function
import sys
sys.path.append('/ebio/ag-neher/share/users/rneher/HIVEVO_access/')
import numpy as np
from itertools import izip
from hivevo.patients import Patient
from hivevo.HIVreference import HIVreference
from hivevo.af_tools import divergence
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns
from scipy.stats import spearmanr, scoreatpercentile
from random import sample
sns.set_style('darkgrid')
fs=16
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

total_muts = {nuc:sum([x for mut, x in nuc_muts.iteritems() if mut[0]==nuc]) for nuc in 'ACGT'}

if __name__=="__main__":
    #patients = ['p1', 'p2','p3','p5','p6', 'p8', 'p9','p10', 'p11'] # all subtypes
    patients = ['p2','p3','p4','p5','p7', 'p8', 'p9','p10', 'p11'] # subtype B only
    regions = ['gag', 'pol']
    subtype='B'
    mut_rate = 2e-5
    cols = sns.color_palette(n_colors=4)
    ls = {'gag':'-', 'pol':'--', 'nef':'-.'}

    hxb2 = HIVreference(refname='HXB2', subtype = 'B')
    good_pos_in_reference = hxb2.get_ungapped(threshold = 0.05)
    cov_min=1000
    combined_af={}
    consensus_mutation_rate={}
    combined_phenos={}
    combined_af_by_pat={}
    weights_by_pat={}
    for region in regions:
        consensus_mutation_rate[region]= np.array([total_muts[x] for x in
                hxb2.annotation[region].extract("".join(hxb2.consensus))])
        combined_af[region] = np.zeros((6, len(hxb2.annotation[region])))
        combined_phenos[region] = np.zeros(len(hxb2.annotation[region]))
        combined_af_by_pat[region] = {}
        weights_by_pat[region] = {}
        for pi, pcode in enumerate(patients):
            combined_af_by_pat[region][pcode] = np.zeros((6, len(hxb2.annotation[region])))

    for pi, pcode in enumerate(patients):
        try:
            p = Patient.load(pcode)
        except:
            print("Can't load patient", pcode)
        else:
            print(pcode, p.Subtype)
            for region in regions:
                aft = p.get_allele_frequency_trajectories(region, cov_min=cov_min, error_rate = 2e-3, type='nuc')

                # get patient to subtype map and subset entropy vectors, convert to bits
                patient_to_subtype = p.map_to_external_reference(region, refname = 'HXB2')
                consensus = hxb2.get_consensus_indices_in_patient_region(patient_to_subtype)
                ancestral = p.get_initial_indices(region)[patient_to_subtype[:,2]]
                rare = ((aft[:,:4,:]**2).sum(axis=1).min(axis=0)>0.5)[patient_to_subtype[:,2]]
                final = aft[-1].argmax(axis=0)[patient_to_subtype[:,2]]
                pp = [p.pos_to_feature[pos]['RNA pairing probability'] for pos in p.annotation[region]]
                pp = np.array(map(lambda x:0.0 if x is None else x, pp))
                combined_phenos[region][patient_to_subtype[:,0]-patient_to_subtype[0][0]]+= pp[patient_to_subtype[:,2]]

                pprob = np.array([p.pos_to_feature[x]['RNA pairing probability']>0.5 for x in p.annotation[region]])
#                plt.figure()
#                print("fraction paired:", pprob.mean())
#                for ci,c in enumerate(p.get_initial_indices(region)):
#                    if pprob[ci]: plt.plot(aft[:,c,ci])

                weights_by_pat[region][pcode]=0
                for af, ysi, depth in izip(aft, p.ysi, p.n_templates_dilutions):
                    if ysi<1:
                        continue
                    pat_af = af[:,patient_to_subtype[:,2]]
                    patient_consensus = pat_af.argmax(axis=0)
                    ind = rare&(patient_consensus==consensus)&(ancestral==consensus)&(final==consensus) #(patient_consensus!=consensus) #&(ancestral==consensus)
                    w = depth/(1+depth/300.0)
                    weights_by_pat[region][pcode]+=w
                    combined_af_by_pat[region][pcode][:,patient_to_subtype[ind,0]-patient_to_subtype[0][0]] \
                                += w*pat_af[:,ind]


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
        minor_af[region] = (combined_af[region][:4,:].sum(axis=0) - combined_af[region].max(axis=0))/(1e-6+combined_af[region][:4,:].sum(axis=0))
        minor_af_bs[region]=[]
        combined_entropy_bs[region]=[]
        #for ii in xrange(nbs):
        #    tmp_af = af_average(patient_bootstrap(combined_af_by_pat[region]))
        #    combined_entropy_bs[region].append((-np.log(tmp_af+1e-10)*tmp_af).sum(axis=0))
        #    minor_af_bs[region].append((tmp_af[:4,:].sum(axis=0) - tmp_af.max(axis=0))/(tmp_af[:4,:].sum(axis=0)+1e-6))
        #minor_af_part[region]=[]
        #combined_entropy_part[region]=[]
        #for ii in xrange(nbs):
        #    for a in patient_partition(combined_af_by_pat[region]):
        #        tmp_af = af_average(a)
        #        combined_entropy_part[region].append((-np.log(tmp_af+1e-10)*tmp_af).sum(axis=0))
        #        minor_af_part[region].append((tmp_af[:4,:].sum(axis=0) - tmp_af.max(axis=0))/(tmp_af[:4,:].sum(axis=0)+1e-6))


    cols = sns.color_palette()
    for region in regions:
        reg = hxb2.annotation[region].location
        xsS = hxb2.entropy[reg.start:reg.end]
        ind = xsS>=0.000
        print(region)
        print(np.corrcoef(combined_entropy[region][ind], xsS[ind]))
        print(spearmanr(combined_entropy[region][ind], xsS[ind]))

        plt.figure()
        #plt.title(region+' entropy scatter', fontsize=fs)
        for ni in range(3):
            ind = (xsS>=0.000)&((np.arange(len(minor_af[region]))%3)==ni)
            plt.scatter(combined_entropy[region][ind]+.00003, xsS[ind]+.005, c=cols[ni], label='position '+str(ni+1), s=30)
        plt.ylabel('cross-sectional entropy', fontsize=fs)
        plt.xlabel('pooled within patient entropy', fontsize=fs)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylim([0.003, 2])
        plt.xlim([0.00001, .3])
        plt.tick_params(labelsize=fs*0.8)
        plt.tight_layout()
        plt.savefig('figures/'+region+'_entropy_scatter.pdf')

#        plt.figure()
#        plt.title(region+' entropy scatter')
#        plt.hexbin(combined_entropy[region][ind]+.00005, xsS[ind]+.012, xscale='log', yscale='log', gridsize=15, cmap = cm.YlOrRd_r, bins='log')
#        plt.ylabel('cross-sectional entropy')
#        plt.xlabel('pooled within patient entropy')


#        plt.figure()
#        plt.title(region+' number of nuc >0')
#        for ni in range(1,5):
#            ind = (combined_af[region]>0.0001).sum(axis=0)==ni
#            y,x = np.histogram(xsS[ind], bins=np.logspace(-2.7,0.3,21))
#            plt.plot(x[1:], y/np.diff(x)/y.sum(), label=str(ni))
#        plt.legend()
#        plt.xscale('log')
#        plt.yscale('log')
#        plt.savefig('figures/'+region+'_entropy_histogram.pdf')

#        plt.figure()
#        plt.title(region+' number of nuc>0')
#        for ni in range(1,5):
#            ind = (combined_af[region]>0.0001).sum(axis=0)==ni
#            y,x = np.histogram(np.arange(len(ind))[ind]%3, bins=[-0.5, 0.5, 1.5, 2.5])
#            plt.plot(1.0*y/y.sum(), label=str(ni)+' n='+str(np.sum(ind)))
#        plt.legend(loc=2)
#        plt.xlabel('codon position')
#        plt.ylabel('fraction of sites')
#        plt.savefig('figures/'+region+'_non_zero_sites.pdf')

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
            plt.plot(sorted(consensus_mutation_rate[region][ind]/(minor_af[region][ind]+0.0001)), np.linspace(1,0,ind.sum()),
                    label='Codon pos:'+str(ni)+' n='+str(np.sum(ind)))
        plt.plot(sorted(consensus_mutation_rate[region]/(minor_af[region]+0.0001)), np.linspace(1,0,len(minor_af[region])),
                label='All:'+str(ni)+' n='+str(len(ind)))
        plt.xscale('log')
        plt.yscale('linear')
        plt.legend(loc=2)
        plt.xlabel('selection coeff')
        plt.ylabel('P(s<X)')
        plt.savefig('figures/'+region+'_sel_coeff.pdf')

        thres = np.arange(0,1.01, 0.2)
        s_coeff = []
        S_cutoff = []
        s_counts = []
        for q_low, q_up in zip(thres[:-1], thres[1:]):
            S_low, S_up = scoreatpercentile(xsS, q_low*100), scoreatpercentile(xsS, q_up*100)
            ind = (xsS>=S_low)&(xsS<S_up)
            tmp_s = [np.median(mut_rate/(minor_af[region]+0.0001)[ind])]
            tmp_Sc = [(S_low+S_up)*0.5]
            tmp_counts = [ind.sum()]
            for ni in range(3):
                pos_ind = (np.arange(len(xsS))%3)==ni
                S_low, S_up = scoreatpercentile(xsS[pos_ind], q_low*100), scoreatpercentile(xsS[pos_ind], q_up*100)
                ind = (xsS[pos_ind]>=S_low)&(xsS[pos_ind]<S_up)
                tmp_s.append(np.median(mut_rate/(minor_af[region][pos_ind][ind]+0.0001)))
                tmp_Sc.append((S_low+S_up)*0.5)
                tmp_counts.append(ind.sum())
            s_counts.append(tmp_counts)
            s_coeff.append(tmp_s)
            S_cutoff.append(tmp_Sc)
            print(q_low, q_up, ":", tmp_s, tmp_counts)

        plt.figure('sel_coeff')
        s_coeff=np.array(s_coeff)
        S_cutoff=np.array(S_cutoff)+.005
        for ii in range(4):
            label = 'codon '+str(ii) if ii else 'all'
            plt.plot(S_cutoff[:,ii], s_coeff[:,ii], 'o-',label=region+': '+label, c=cols[ii], ls=ls[region])
        plt.yscale('log')
        plt.xscale('log')
        plt.legend(loc=3, numpoints=2)
        plt.ylabel('fitness cost')
        plt.xlabel('conservation')

        #ii=4
        #plt.figure()
        #ind = (~(np.isnan(minor_af_part[region][ii])|np.isnan(minor_af_part[region][ii+1])))&\
        #      ((minor_af_part[region][ii]>0) & (minor_af_part[region][ii+1]>0))
        #plt.scatter(minor_af_part[region][ii][ind]+1e-6, minor_af_part[region][ii+1][ind]+1e-6) #, yscale='log', xscale='log')
        #print(spearmanr(minor_af_part[region][ii][ind]+1e-6, minor_af_part[region][ii+1][ind]+1e-6))
        #plt.yscale('log')
        #plt.xscale('log')
        #plt.xlim([1e-6,1])
        #plt.ylim([1e-6,1])
#
        # ii=2
        # plt.figure()
        # plt.hist(np.log10(minor_af_part[region][ii][ind]+1e-6) - np.log10(minor_af_part[region][ii+1][ind]+1e-6), bins=30)
        # print("Difference std:", np.std(np.log10(minor_af_part[region][ii][ind]+1e-6) - np.log10(minor_af_part[region][ii+1][ind]+1e-6)))
        # print("Single std:", np.std(np.log10(minor_af_part[region][ii][ind]+1e-6)), np.std(np.log10(minor_af_part[region][ii+1][ind]+1e-6)))


        # plt.figure()
        # minor_af_array=np.array(minor_af_bs[region])

        # plt.title(region+' min/max frequency of boot strap replicates')
        # minor_af_extrema = np.vstack((minor_af_array.min(axis=0), np.median(minor_af_array, axis=0), minor_af_array.max(axis=0)))
        # sorted_by_median = np.array(sorted(minor_af_extrema.T, key=lambda x:x[1]))
        # plt.plot(sorted_by_median[:,0], label='minimum')
        # plt.plot(sorted_by_median[:,1], label='median')
        # plt.plot(sorted_by_median[:,2], label='maximum')
        # plt.yscale('log')
        # plt.ylabel('frequency')
        # plt.xlabel('positions sorted by median')
        # plt.legend(loc=2)
        # plt.savefig('figures/'+region + '_frequency_noise.pdf')

        fig, axs = plt.subplots(1,3,sharey=True)
        #plt.title(region+' selection coefficients')
        for ni,ax in enumerate(axs):
            ind = (np.arange(len(minor_af[region]))%3)==ni
            s = consensus_mutation_rate[region][ind]/(minor_af[region][ind]+0.0001)
            s[s>=0.1] = 0.1
            s[s<=0.001] = 0.001
            ax.hist(s, color=cols[ni],
                     weights=np.ones(ind.sum())/ind.sum(), bins=np.logspace(-3,-1,11), label='position: '+str(ni))
            ax.set_xscale('log')
            ax.tick_params(labelsize=fs*0.8)
            ax.text(0.1, 0.8, 'position: '+str(ni))
            if ni==0:
                ax.set_ylabel('P(s)', fontsize=fs)
                ax.set_yscale('linear')
            if ni==1: ax.set_xlabel('selection coefficient', fontsize=fs)
        plt.tight_layout()
        plt.savefig('figures/'+region+'_selection_coefficients.pdf')
