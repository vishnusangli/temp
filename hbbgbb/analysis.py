import numpy as np
import matplotlib.pyplot as plt

import kkplot

from . import plot as myplt

def roc(df, score, output=None, plot=True):
    """
    Create ROC curves given `score` column.

    The return value is a dictionary with key `label#` and value the CDF of the
    score for that label. 

    Optional output is also supported as npy (`output="fileprefix"`) or
    plots (`plot==True`, saved to `output`).
    """
    labels=sorted(df['label'].unique())

    # Calculate ROC curves
    rocs={}

    mymin=np.floor(df[score].min())
    mymax=np.ceil(df[score].max())

    for label in labels:
        h,b=np.histogram(df[df.label==label][score],bins=100,range=(mymin,mymax))
        h=1-np.cumsum(h)/np.sum(h) # turn into CDF
        rocs[label]=h

    # Plot ROC curves
    if plot:
        fig, ax=plt.subplots(1,1,figsize=(8,6))
        for label in labels:
            if label==0: continue # this is signal
            ax.plot(rocs[0],1-rocs[label],'-',label=myplt.mylabels.get(label,label))
    
        ax.set_xlabel('Signal Efficiency')
        ax.set_ylabel('Background Rejection')
        kkplot.ticks(ax.xaxis, 0.1, 0.02)
        kkplot.ticks(ax.yaxis, 0.1, 0.02)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        fig.legend(title='Background')
        fig.tight_layout()
        if output is not None:
            fig.savefig(output)
        fig.show()

    # Save ROC curves
    if output is not None:
        np.save(f'{output}.npy',rocs)
