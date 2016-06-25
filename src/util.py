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
    fn = '../data/mutation_rates/mutation_rate_'+str(threshold) + ('_gp120' if gp120 else '') + '.pickle'
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



def draw_genome(annotations,
                ax=None,
                rows=4,
                readingframe=True, fs=9,
                y1=0,
                height=1,
                pad=0.2):
    '''Draw genome boxes'''
    from matplotlib.patches import Rectangle
    from Bio.SeqFeature import CompoundLocation
    import pandas as pd

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.set_ylim([-pad,rows*(height+pad)])
    anno_elements = []
    for name, feature in annotations.iteritems():
        if type(feature.location) is CompoundLocation:
            locs = feature.location.parts
        else:
            locs = [feature.location]
        for li,loc in enumerate(locs):
            x = [loc.nofuzzy_start, loc.nofuzzy_end]
            anno_elements.append({'name': name,
                                  'x1': x[0],
                                  'x2': x[1],
                                  'width': x[1] - x[0]})
            if name[0]=='V':
                anno_elements[-1]['ri']=3
            elif li:
                anno_elements[-1]['ri']=(anno_elements[-2]['ri'] + ((x[0] - anno_elements[-2]['x2'])%3))%3
            else:
                anno_elements[-1]['ri']=x[0]%3

    anno_elements.sort(key = lambda x:x['x1'])
    for ai, anno in enumerate(anno_elements):
        if readingframe:
            anno['y1'] = y1 + (height + pad) * anno['ri']
        else:
            anno['y1'] = y1 + (height + pad) * (ai%rows)
        anno['y2'] = anno['y1'] + height
        anno['height'] = height

    for anno in anno_elements:
        r = Rectangle((anno['x1'], anno['y1']),
                      anno['width'],
                      anno['height'],
                      facecolor=[0.8] * 3,
                      edgecolor='k',
                      label=anno['name'])

        xt = anno['x1'] + 0.5 * anno['width']
        yt = anno['y1'] + 0.2 * height + height * (anno['width']<500)
        anno['x_text'] = xt
        anno['y_text'] = yt

        ax.add_patch(r)
        ax.text(xt, yt,
                anno['name'],
                color='k',
                fontsize=fs,
                ha='center')

    return pd.DataFrame(anno_elements)

