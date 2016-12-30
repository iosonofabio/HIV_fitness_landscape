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
    for gene in genes:
        fc = pd.read_csv('../data/fitness_pooled/nuc_'+gene+'_selection_coeffcients_'+subtype+'.tsv', sep='\t', header=1)
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
        for nuc in 'ACGT':
            ind = np.array(cons==nuc)&(~np.isnan(fitness_array))
            #plt.hist(fitness_array[ind], bins=np.logspace(-3,-1,21), weights=1.0/ind.sum()*np.ones(ind.sum()))
            plt.plot(sorted(fitness_array[ind]), np.linspace(0,1,ind.sum()), label=nuc)
        plt.xscale('log')
        plt.legend(loc=2)

        plt.figure()
        plt.title(gene)
        cpg = np.zeros_like(fitness_array, dtype='bool')
        cpg[:-1] = cpg[:-1]|((cons[:-1]=='T')&(cons[1:]=='G'))
        cpg[1:] = cpg[1:]|((cons[:-1]=='C')&(cons[1:]=='A'))

        gpc = np.zeros_like(fitness_array, dtype='bool')
        gpc[:-1] = gpc[:-1]|((cons[:-1]=='G')&(cons[1:]=='T'))
        gpc[1:] = gpc[1:]|((cons[:-1]=='A')&(cons[1:]=='C'))
        #cpg[:-1] = cpg[:-1]|((cons[:-1]=='G')&(cons[1:]=='T'))
        #cpg[1:] = cpg[1:]|((cons[:-1]=='A')&(cons[1:]=='C'))
        cpg = cpg&(~np.isnan(fitness_array))
        not_cpg = (gpc)&(~np.isnan(fitness_array))
        #plt.hist(fitness_array[ind], bins=np.logspace(-3,-1,21), weights=1.0/ind.sum()*np.ones(ind.sum()))
        plt.plot(sorted(fitness_array[cpg]), np.linspace(0,1,cpg.sum()), label='to CpG, n='+str(cpg.sum()))
        plt.plot(sorted(fitness_array[not_cpg]), np.linspace(0,1,(not_cpg).sum()), label='not to CpG, n='+str(not_cpg.sum()))
        print('KS: ', ks_2samp(fitness_array[cpg], fitness_array[not_cpg]),
              "to cpg: %1.4f, not: %1.4f"%(np.mean(fitness_array[cpg]), np.mean(fitness_array[not_cpg])))
        plt.xscale('log')
        plt.legend(loc=2)

