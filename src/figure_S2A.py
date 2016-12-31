# vim: fdm=indent
'''
author:     Fabio Zanini
date:       30/12/16
content:    Figure S2A on sensitivity of the mutation rate estimates to the
            entropy threshold and env/non-env.
'''
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

fs=16

def load_mutation_rate(fname):
    from cPickle import load
    return load(open(fname))

if __name__=="__main__":
    Sthres = np.array([0.01, 0.03, 0.1, 0.3, 0.5])
    fig=plt.figure()
    ax = plt.subplot(111)
    for ex in [[],['gp120']]:
        muts = defaultdict(list)
        for thres in Sthres:
            mut = load_mutation_rate('../data/mutation_rate_'+'_'.join([str(thres)]+ex)+'.pickle')
            for m in mut.iterrows():
                muts[m[0]].append((m[1]['mu'], m[1]['dmulog10']))
        for mi, m in enumerate(sorted(muts.keys())):
            y = [x[0] for x in muts[m]]
            yerr = [np.exp(np.log(10)*x[1])*x[0] for x in muts[m]]
            ax.plot(1.015**mi*Sthres, y, label=m if len(ex) else None,
                    ls='-' if len(ex) else '--', marker='o' if mi<6 else 'v')
#            ax.errorbar(1.015**mi*Sthres, y, yerr, label=m if len(ex) else None,
#                        ls='-' if len(ex) else '--')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('rate', fontsize = fs)
    ax.set_xlabel('diversity threshold', fontsize = fs)
    ax.tick_params(labelsize=0.8*fs)
    ax.legend(ncol=3, fontsize = 0.8*fs)
    ax.set_ylim(1e-8,1e-3)
    plt.savefig('../figures/threshold_sensitivity.pdf')
    plt.savefig('../figures/figure_S2A.pdf')
