import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from rank_correlation import parse_results


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
    YERR = std / np.sqrt(n)
    x = mean.index.values

    fig, ax = plt.subplots(figsize=figsize)
    # plt.subplots_adjust(left=0.1, right=0.72, bottom=0.15, top=0.98)
    ax.errorbar(
        x,
        mean.ov_db.values,
        yerr=YERR.ov_db.values,
        fmt='o',
        ms=4,
        elinewidth=0.5,
        label='$C_B$')
    ax.errorbar(
        x,
        mean.ov_dc.values,
        yerr=YERR.ov_dc.values,
        fmt='o',
        ms=4,
        elinewidth=0.5,
        label='$C_C$')
    ax.errorbar(
        x,
        mean.ov_de.values,
        yerr=YERR.ov_de.values,
        fmt='o',
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
        fontsize='small')
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
    import pdb
    pdb.set_trace()
    egn = offset_ccdf(df, 'offset_egn')

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(btn.index.values, btn.values, label='$C_B$')
    ax.plot(cln.index.values, cln.values, label='$C_C$')
    ax.plot(egn.index.values, egn.values, label='$C_E$')
    ax.set_xlabel('Ranking position of the head of $C_D$, $x$')
    ax.set_ylabel('$Pr(X>=x)$')
    ax.legend()
    plt.tight_layout(pad=0.2)
    plt.savefig(output)


def centrality_rank_correlation(size=20, md2=2):
    output = 'ba-centrality-rank-correlation-md-{}.pdf'.format(md2)

    df = parse_results()
    import ipdb; ipdb.set_trace()
    df = df.groupby(['N', 'md', 'model']).head(size)

    for col in [
            'db_k', 'db_p', 'db_s', 'dc_k', 'dc_p', 'dc_s', 'de_k', 'de_p',
            'de_s', 'r_a', 'rho_a', 'tau_a'
    ]:
        df.loc[:, col] = df[col].apply(lambda x: x[0])
    df = df.sort_values('N')
    gps = df.groupby(['md', 'model'])
    gp0 = gps.get_group((1, 'pa'))
    gp1 = gps.get_group((md2, 'pa'))
    gp2 = gps.get_group((1, 'configuration'))
    gp3 = gps.get_group((md2, 'configuration'))

    fig, axes = plt.subplots(4, 3, figsize=(8, 9), sharex=True, sharey=True)
    r0 = 0
    c0 = 0
    X = df.N.unique()

    def cell_err_bar(ax, gp, coef_pre='db_', legend=False):
        colors = ['#fc8d59', '#ffffbf', '#91bfdb']
        ebs = []
        for post, color, marker in zip(['p', 's', 'k'], colors,
                                       ['o', 's', 'd']):
            # import ipdb; ipdb.set_trace()
            if post == 'p':
                label = r"Pearson's $r$"
            elif post == 's':
                label = r"Spearman's $\rho$"
            elif post == 'k':
                label = r"Kendall's $\tau$"
            Y_data = gp[coef_pre + post].values.reshape(-1, size)
            Y = Y_data.mean(axis=1)
            Y_e = Y_data.std(ddof=1, axis=1)
            eb = ax.errorbar(
                X,
                Y,
                yerr=Y_e,
                label=label,
                fmt='o',
                ms=4,
                # marker=marker,
                # mfc=color,
                # mec=color,
                # ecolor=color,
                alpha=0.8)
            ebs.append(eb)
        ax.set_xscale('log')
        if legend is True:
            ax.legend(handles=ebs, fontsize='small')

    cell_err_bar(axes[r0 + 0, c0 + 0], gp0, 'db_', legend=True)
    cell_err_bar(axes[r0 + 0, c0 + 1], gp0, 'dc_')
    cell_err_bar(axes[r0 + 0, c0 + 2], gp0, 'de_')

    cell_err_bar(axes[r0 + 1, c0 + 0], gp1, 'db_')
    cell_err_bar(axes[r0 + 1, c0 + 1], gp1, 'dc_')
    cell_err_bar(axes[r0 + 1, c0 + 2], gp1, 'de_')

    cell_err_bar(axes[r0 + 2, c0 + 0], gp2, 'db_')
    cell_err_bar(axes[r0 + 2, c0 + 1], gp2, 'dc_')
    cell_err_bar(axes[r0 + 2, c0 + 2], gp2, 'de_')

    cell_err_bar(axes[r0 + 3, c0 + 0], gp3, 'db_')
    cell_err_bar(axes[r0 + 3, c0 + 1], gp3, 'dc_')
    cell_err_bar(axes[r0 + 3, c0 + 2], gp3, 'de_')

    axes[r0 + 3, c0 + 1].set_xlabel('Network Size, $N$')
    axes[r0 + 1, c0 + 0].set_ylabel('Coefficient')
    bbox_props_h = dict(
        boxstyle='rarrow, pad=1.2',
        mutation_aspect=1.2,
        fc=None,
        ec='cyan',
        lw=0.6,
        alpha=0.15)
    bbox_props_v = dict(
        boxstyle='larrow, pad=1.2',
        mutation_aspect=1.5,
        fc=None,
        ec='cyan',
        lw=0.6,
        alpha=0.15)

    def set_cell_of_1st_row(ax, label):
        # ax.axis('off')
        ax.text(
            0.5,
            1.18,
            ' ',
            ha='center',
            va='bottom',
            rotation=90,
            size=12,
            bbox=bbox_props_v,
            transform=ax.transAxes)
        ax.text(
            0.48,
            1.15,
            label,
            ha='center',
            va='bottom',
            size=9,
            transform=ax.transAxes)

    def set_cell_of_1st_column(ax, label):
        # ax.axis('off')
        ax.text(
            -0.5,
            0.5,
            '        ',
            ha='right',
            va='center',
            rotation=0,
            size=12,
            bbox=bbox_props_h,
            transform=ax.transAxes)
        ax.text(
            -0.38,
            0.5,
            label,
            ha='right',
            va='center',
            size=9,
            transform=ax.transAxes)

    set_cell_of_1st_row(axes[r0 + 0, c0 + 0], '$corr(D, B)$')
    set_cell_of_1st_row(axes[r0 + 0, c0 + 1], '$corr(D, C)$')
    set_cell_of_1st_row(axes[r0 + 0, c0 + 2], '$corr(D, E)$')
    set_cell_of_1st_column(axes[r0 + 0, c0], 'BA, $\Delta m=1$')
    set_cell_of_1st_column(axes[r0 + 1, c0], 'BA, $\Delta m={}$'.format(md2))
    set_cell_of_1st_column(axes[r0 + 2, c0], 'BAC, $\Delta m=1$')
    set_cell_of_1st_column(axes[r0 + 3, c0], 'BAC, $\Delta m={}$'.format(md2))
    plt.tight_layout(rect=[0.12, 0, 1.02, 0.95])
    plt.savefig(output)


def centrality_rank_correlation_2line(size=20):
    output = 'ba-centrality-rank-correlation.pdf'

    df = parse_results()
    df = df.groupby(['N', 'md', 'model']).head(size)

    for col in [
            'db_k', 'db_p', 'db_s', 'dc_k', 'dc_p', 'dc_s', 'de_k', 'de_p',
            'de_s', 'r_a', 'rho_a', 'tau_a'
    ]:
        df.loc[:, col] = df[col].apply(lambda x: x[0])
    df = df.sort_values('N')
    gps = df.groupby(['md', 'model'])
    gp0 = gps.get_group((1, 'pa'))
    gp1 = gps.get_group((10, 'pa'))
    gp2 = gps.get_group((1, 'configuration'))
    gp3 = gps.get_group((10, 'configuration'))

    fig, axes = plt.subplots(4, 3, figsize=(8, 6), sharex=True, sharey=True)
    r0 = 0
    c0 = 0
    X = df.N.unique()

    def cell_err_bar(ax, gp, coef_pre='db_'):
        for post, color in zip(['p', 's', 'k'], ['r', 'g', 'b']):
            # import ipdb; ipdb.set_trace()
            Y_data = gp[coef_pre + post].values.reshape(-1, size)
            Y = Y_data.mean(axis=1)
            Y_e = Y_data.std(ddof=1, axis=1)
            ax.plot(X, Y, color=color)
            ax_ = ax.twinx()
            ax_.plot(X, Y_e, color=color)
        ax.set_xscale('log')

    cell_err_bar(axes[r0 + 0, c0 + 0], gp0, 'db_')
    cell_err_bar(axes[r0 + 0, c0 + 1], gp0, 'dc_')
    cell_err_bar(axes[r0 + 0, c0 + 2], gp0, 'de_')

    cell_err_bar(axes[r0 + 1, c0 + 0], gp1, 'db_')
    cell_err_bar(axes[r0 + 1, c0 + 1], gp1, 'dc_')
    cell_err_bar(axes[r0 + 1, c0 + 2], gp1, 'de_')

    cell_err_bar(axes[r0 + 2, c0 + 0], gp2, 'db_')
    cell_err_bar(axes[r0 + 2, c0 + 1], gp2, 'dc_')
    cell_err_bar(axes[r0 + 2, c0 + 2], gp2, 'de_')

    cell_err_bar(axes[r0 + 3, c0 + 0], gp3, 'db_')
    cell_err_bar(axes[r0 + 3, c0 + 1], gp3, 'dc_')
    cell_err_bar(axes[r0 + 3, c0 + 2], gp3, 'de_')

    axes[r0 + 3, c0 + 1].set_xlabel('Network Size, $N$')
    axes[r0 + 1, c0 + 0].set_ylabel('Coefficient')
    bbox_props_h = dict(
        boxstyle='rarrow, pad=1.2',
        mutation_aspect=1.2,
        fc=None,
        ec='cyan',
        lw=0.6,
        alpha=0.15)
    bbox_props_v = dict(
        boxstyle='larrow, pad=1.2',
        mutation_aspect=1.5,
        fc=None,
        ec='cyan',
        lw=0.6,
        alpha=0.15)

    def set_cell_of_1st_row(ax, label):
        # ax.axis('off')
        ax.text(
            0.5,
            1.18,
            ' ',
            ha='center',
            va='bottom',
            rotation=90,
            size=12,
            bbox=bbox_props_v,
            transform=ax.transAxes)
        ax.text(
            0.48,
            1.15,
            label,
            ha='center',
            va='bottom',
            size=9,
            transform=ax.transAxes)

    def set_cell_of_1st_column(ax, label):
        # ax.axis('off')
        ax.text(
            -0.5,
            0.5,
            '        ',
            ha='right',
            va='center',
            rotation=0,
            size=12,
            bbox=bbox_props_h,
            transform=ax.transAxes)
        ax.text(
            -0.38,
            0.5,
            label,
            ha='right',
            va='center',
            size=9,
            transform=ax.transAxes)

    set_cell_of_1st_row(axes[r0 + 0, c0 + 0], '$corr(D, B)$')
    set_cell_of_1st_row(axes[r0 + 0, c0 + 1], '$corr(D, C)$')
    set_cell_of_1st_row(axes[r0 + 0, c0 + 2], '$corr(D, E)$')
    set_cell_of_1st_column(axes[r0 + 0, c0], 'BA, $\Delta m=1$  ')
    set_cell_of_1st_column(axes[r0 + 1, c0], 'BA, $\Delta m=10$ ')
    set_cell_of_1st_column(axes[r0 + 2, c0], 'BAC, $\Delta m=1$ ')
    set_cell_of_1st_column(axes[r0 + 3, c0], 'BAC, $\Delta m=10$')
    plt.tight_layout(rect=[0.12, 0, 1.02, 0.95])
