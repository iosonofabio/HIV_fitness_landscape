from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from scipy.stats import ks_2samp
sns.set_style('darkgrid')

blossum62={
    'A':4,'R':5,'N':6,'D':6,'C':9,'Q':5,'E':5,'G':6,'H':8,'I':4,
    'L':4,'K':5,'M':5,'F':6,'P':7,'S':4,'T':5,'W':11,'Y':7,'V':4,
}
blossum80={
    'A':7,'R':9,'N':9,'D':10,'C':13,'Q':9,'E':8,'G':9,'H':12,'I':7,
    'L':6,'K':8,'M':9,'F':10,'P':12,'S':7,'T':8,'W':16,'Y':11,'V':7,}

if __name__=="__main__":
    plt.ion()
    genes = ['gag', 'pol', 'env']
    subtype = 'B'
    ls= ['-', '--', '-.', ':']
    fs=16
    alpha_aa='ARNDCQEGHILKMFPSTWYV'
    fraction_lethal = {}
    for gene in genes:
        fc = pd.read_csv('../data/fitness_pooled_aa/aa_'+gene+'_fitness_costs_st_'+subtype+'.tsv', sep='\t', header=1)
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


    fig, axs = plt.subplots(1,3, figsize=(12,5))
    ai = 0
    for gi, gene1 in enumerate(genes):
        for gene2 in genes[gi+1:]:
            ax=axs.flatten()[ai]
            ax.scatter([fraction_lethal[gene1][aa][0] for aa in alpha_aa],
                       [fraction_lethal[gene2][aa][0] for aa in alpha_aa],
                       c=[blossum80[aa] for aa in alpha_aa], cmap=cm.RdBu_r, s=50)
            for aa in alpha_aa:
                ax.annotate(aa, (fraction_lethal[gene1][aa][0]+0.02, fraction_lethal[gene2][aa][0]))
            ax.set_xlabel('fraction s>0.05 ' + gene1, fontsize=fs)
            ax.set_ylabel('fraction s>0.05 ' + gene2, fontsize=fs)
            ax.tick_params(labelsize=0.8*fs)
            ai+=1
            ax.set_xlim(0,1.1)
            ax.set_ylim(0,1.1)
    plt.tight_layout()
    plt.savefig('../figures/fraction_lethal.pdf')
