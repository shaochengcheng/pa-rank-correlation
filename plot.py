import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def overlapping(fn='data/overlapping.csv'):
    output = 'r-intersection.pdf'
    figsize = (4, 3)
    df = pd.read_csv(fn, index_col=0)
    df = df.drop('fn', axis=1)
    gp = df.groupby('threshold')
    mean = gp.mean()
    std = gp.std()
    n = gp.apply(lambda x: x.notnull().sum())
    n = n.drop('threshold', axis=1)
    YERR = std/np.sqrt(n)
    x = mean.index.values

    fig, ax = plt.subplots(figsize=figsize)
    # plt.subplots_adjust(left=0.1, right=0.72, bottom=0.15, top=0.98)
    ax.errorbar(x, mean.ov_db.values, yerr=YERR.ov_db.values, fmt='o',
                ms=4,
                elinewidth=0.5,
              label='$C_B$')
    ax.errorbar(x, mean.ov_dc.values, yerr=YERR.ov_dc.values, fmt='o',
                ms=4,
                elinewidth=0.5,
              label='$C_C$')
    ax.errorbar(x, mean.ov_de.values, yerr=YERR.ov_de.values, fmt='o',
                ms=4,
                elinewidth=0.5,
              label='$C_E$')
    ax.set_ylim([0.1, 1.1])
    ax.set_xscale('log')
    ax.set_xlabel('Ratio of top ranking')
    ax.set_ylabel('Ratio of intersection')
    ax.legend(
        #bbox_to_anchor=(1.02, 1),
        #loc=2,
        #borderaxespad=0.
        fontsize='small'
    )
    plt.tight_layout(pad=0.2)
    plt.savefig(output)


def ccdf(s):
    """
    Parameters:
        `s`, series, the values of s should be variable to be handled
    Return:
        a new series `s`, index of s will be X axis (number), value of s
        will be Y axis (probability)
    """
    s = s.dropna()
    s = s.sort_values(ascending=True, inplace=False)
    s.reset_index(drop=True, inplace=True)
    n = len(s)
    s.drop_duplicates(keep='first', inplace=True)
    X = s.values
    Y = [n - i for i in s.index]

    return pd.Series(data=Y, index=X) / n


def offset_ccdf(df, column):
    offset = (df[column] - 1) / (df['vnum'] - 1)
    return ccdf(offset)


def plot_offset_ccdf(fn='data/offset.csv'):
    figsize = (4, 3)
    output = 'r-offset-ccdf.pdf'
    df = pd.read_csv(fn, index_col=0)

    btn = offset_ccdf(df, 'offset_btn')
    cln = offset_ccdf(df, 'offset_cln')
    import pdb; pdb.set_trace()
    egn = offset_ccdf(df, 'offset_egn')

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(btn.index.values, btn.values,
            label='$C_B$')
    ax.plot(cln.index.values, cln.values,
            label='$C_C$')
    ax.plot(egn.index.values, egn.values,
            label='$C_E$')
    ax.set_xlabel('Ranking position of the head of $C_D$, $x$')
    ax.set_ylabel('$Pr(X>=x)$')
    ax.legend()
    plt.tight_layout(pad=0.2)
    plt.savefig(output)







