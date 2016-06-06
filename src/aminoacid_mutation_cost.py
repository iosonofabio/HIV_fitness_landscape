from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
sns.set_style('darkgrid')

if __name__=="__main__":
    plt.ion()
    genes = ['gag', 'pol', 'env', 'nef']
    subtype = 'B'
    ls= ['-', '--', '-.', ':']
    fs=16
    alpha_aa='ARNDCQEGHILKMFPSTWYV'
    fraction_lethal = {}
    for gene in genes:
        fc = pd.read_csv('../data/'+gene+'_selection_coefficients_st_'+subtype+'.tsv', sep='\t', header=1)
        cons = np.array(fc.loc[:,'consensus'])
        fitness = fc.loc[:,'median']
        fitness_array = []
        for f in fitness:
            if f=='>0.1':
                fitness_array.append(0.1)
            elif f=='<0.001':
                fitness_array.append(0.001)
            else:
                fitness_array.append(float(f))
        fitness_array = np.array(fitness_array)
        plt.figure()
        plt.title(gene)
        fraction_lethal[gene] = {}
        for ai, aa in enumerate(alpha_aa):
            ind = np.array(cons==aa)&(~np.isnan(fitness_array))
            fraction_lethal[gene][aa] = (np.mean(fitness_array[ind]>0.05),str(ind.sum()))
            #plt.hist(fitness_array[ind], bins=np.logspace(-3,-1,21), weights=1.0/ind.sum()*np.ones(ind.sum()))
            plt.plot(sorted(fitness_array[ind]), np.linspace(0,1,ind.sum()),ls=ls[ai//6], label=aa +' n='+str(ind.sum()))
        print(gene)
        for aa,(f,n) in sorted(fraction_lethal[gene].items(), key=lambda x:x[1][0]):
            print(aa,np.round(f,2), n)
        plt.xscale('log')
        plt.legend(loc=2, ncol=3)


    fig, axs = plt.subplots(2,3, figsize=(12,8))
    ai = 0
    for gi, gene1 in enumerate(genes):
        for gene2 in genes[gi+1:]:
            ax=axs.flatten()[ai]
            ax.scatter([fraction_lethal[gene1][aa][0] for aa in alpha_aa], [fraction_lethal[gene2][aa][0] for aa in alpha_aa])
            for aa in alpha_aa:
                ax.annotate(aa, (fraction_lethal[gene1][aa][0]+0.02, fraction_lethal[gene2][aa][0]))
            ax.set_xlabel(gene1, fontsize=fs)
            ax.set_ylabel(gene2, fontsize=fs)
            ax.tick_params(labelsize=0.8*fs)
            ai+=1
            ax.set_xlim(0,1.1)
            ax.set_ylim(0,1.1)
    plt.tight_layout()
    plt.savefig('../figures/fraction_lethal.pdf')
