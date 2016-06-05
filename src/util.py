# vim: fdm=indent
'''
author:     Fabio Zanini
date:       01/12/15
content:    Utility functions for the HIV fitness landscape project.
'''
# Modules
import numpy as np



# Globals
fig_width = 5
fig_fontsize = 12


# Functions
def load_mutation_rates(threshold=0.3, gp120=True):
    import pandas as pd
    fn = '../data/mutation_rate_'+str(threshold) + ('_gp120' if gp120 else '') + '.pickle'
    print('loading', fn)
    mu =  pd.read_pickle(fn)
    return mu


# Functions
def add_binned_column(df, bins, to_bin):
    '''Add a column to data frame with binned values (in-place)

    Parameters
       df (pandas.DataFrame): data frame to change in-place
       bins (array): bin edges, including left- and rightmost
       to_bin (string): prefix of the new column name, e.g. 'A' for 'A_bin'

    Returns
       None: the column are added in place
    '''
    # FIXME: this works, but is a little cryptic
    df.loc[:, to_bin+'_bin'] = np.minimum(len(bins)-2,
                                          np.maximum(0,np.searchsorted(bins, df.loc[:,to_bin])-1))


def boot_strap_patients(df, eval_func, columns=None,  n_bootstrap=100):
    import pandas as pd

    if columns is None:
        columns = df.columns
    if 'pcode' not in columns:
        columns = list(columns)+['pcode']

    patients = df.loc[:,'pcode'].unique()
    tmp_df_grouped = df.loc[:,columns].groupby('pcode')
    npats = len(patients)
    replicates = []
    for i in xrange(n_bootstrap):
        if (i%20==0): print("Bootstrap",i)
        pats = patients[np.random.randint(0,npats, size=npats)]
        bs = []
        for pi,pat in enumerate(pats):
            bs.append(tmp_df_grouped.get_group(pat))
            bs[-1]['pcode']='BS'+str(pi+1)
        bs = pd.concat(bs)
        replicates.append(eval_func(bs))
    return replicates


def add_panel_label(ax, label, x_offset=-0.1):
    '''Add a label letter to a panel'''
    ax.text(x_offset, 0.95, label,
            transform=ax.transAxes,
            fontsize=fig_fontsize*1.5)
