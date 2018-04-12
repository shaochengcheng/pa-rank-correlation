import graph_tool.all as gt
import numpy as np
import pandas as pd
from scipy import stats
from multiprocessing import Process, Manager
from queue import Empty
import pickle
import time
import parse
import logging
import sys
from os.path import join
from os import listdir
from pathlib import Path

logger = logging.getLogger(__name__)
FILENAME_PATTERN = 'n_{}_md_{}_{}_{}.pkl'
N =np.logspace(10, 20, 11, base=2).astype(int)
MD=[1, 10]
MODELS = ['pa', 'configuration']


def gen_filename(n, md, model, data_dir='pkls'):
    filename = FILENAME_PATTERN.format(n, md, model, int(time.time()))
    return join(data_dir, filename)

def parse_filename(filename):
    filename = Path(filename).resolve().stem
    n, md, model, t = parse.parse(FILENAME_PATTERN, filename)
    return (int(n), int(md), model, int(t))

def parse_results(data_dir='pkls', validate_num_keys=17):
    r = []
    for filename in listdir(data_dir):
        filename = join(data_dir, filename)
        if filename.endswith('.pkl'):
            with open(filename, 'rb') as f:
                loaded_r = pickle.load(f)
            if len(loaded_r) == validate_num_keys:
                r.append(loaded_r)
            else:
                logger.warning('Insufficient number of keys of results: %s',
                        filename)
        else:
            logger.warning('Note a pickle file, should end with .pkl')
    df = pd.DataFrame(r)
    df['N'] = df.N.astype(int)
    return df


def half_swap(df):
    cols = list(df.columns)
    r_cols = [cols[1], cols[0]]
    half_i = int(len(df) / 2.0)
    df1 = df.loc[:half_i]
    df2 = df.loc[half_i + 1:]
    # swap df1
    df1 = df1[r_cols]
    df1.columns = cols
    return pd.concat([df1, df2], ignore_index=True)


def rank_assortativity(g, method='spearman', n=10):
    # import pdb; pdb.set_trace()
    edges = g.get_edges()
    # shuffle the edges
    df = pd.DataFrame(edges, columns=['s', 't', 'i'])
    df = df[['s', 't']]
    r = []
    for i in range(n):
        df = df.sample(frac=1).reset_index(drop=True)
        df = half_swap(df)
        dgr = g.degree_property_map('total').a
        df = df.applymap(lambda v: dgr[v])
        a = df['s'].values
        b = df['t'].values
        if method == 'spearman':
            c, p = stats.spearmanr(a, b)
            r.append(c)
        elif method == 'kendall':
            c, p = stats.kendalltau(a, b)
            r.append(c)
        else:
            raise TypeError('Unknown method')
    return np.mean(r), np.std(r, ddof=1) / np.sqrt(n)


def average_path(g):
    dist = gt.shortest_distance(g, directed=False)
    n = g.num_vertices()
    return sum([sum(v) for v in dist]) / (n * (n - 1))


def g_centrality_correlations(g):
    dgr = g.degree_property_map('total').a
    dgr = dgr / (g.num_vertices() - 1)
    btn = gt.betweenness(g, norm=True)[0].a
    cln = gt.closeness(g, norm=True, harmonic=False).a
    egn = gt.eigenvector(g)[1].a
    return dict(
        # dgr=dgr,
        # btn=btn,
        # cln=cln,
        # egn=egn,
        db_p=stats.pearsonr(dgr, btn),
        dc_p=stats.pearsonr(dgr, cln),
        de_p=stats.pearsonr(dgr, egn),
        db_s=stats.spearmanr(dgr, btn),
        dc_s=stats.spearmanr(dgr, cln),
        de_s=stats.spearmanr(dgr, egn),
        db_k=stats.kendalltau(dgr, btn),
        dc_k=stats.kendalltau(dgr, cln),
        de_k=stats.kendalltau(dgr, egn))


def g_one_main(n, md, model, to_pickle=True, data_dir='pkls'):
    r = dict()
    g = gt.price_network(
        N=n,
        m=md,
        gamma=1,
        directed=False,
        seed_graph=gt.complete_graph(N=md+1))
    if model == 'configuration':
        gt.random_rewire(g, model='configuration')
    r['N'] = n
    r['md'] = md
    r['model'] = model
    try:
        r.update(dict(
            n=g.num_vertices(),
            m=g.num_edges(),
            # l=average_path(g),
            # cc=gt.global_clustering(g)
            r_a=gt.assortativity(g, 'total'),
            rho_a=rank_assortativity(g, method='spearman'),
            tau_a=rank_assortativity(g, method='kendall')
            ) )
        r.update(g_centrality_correlations(g))
    except Exception as e:
        logger.error(e)
    if to_pickle is True:
        filename = gen_filename(n, md, model, data_dir)
        with open(filename, 'wb') as f:
            pickle.dump(r, f, -1)
    return r


def pa_main(nround=10):
    for r in range(nround):
        for n in N:
            for md in MD:
                for model in MODELS:
                    logger.info('Current graph settings: N=%s, md=%r, model=%s',
                            n, md, model)
                    g_one_main(n, md, model)


def gen_bigred2_batch(filename='bg2_batch_params.txt',
            N=np.logspace(21, 24, 4, base=2).astype(int),
            nrounds=10):
    with open(filename, 'w') as f:
        for r in range(nrounds):
            for n in N:
                for md in MD:
                    for model in MODELS:
                        f.write('python rank_correlation.py {} {} {}\n'.format(
                            n, md, model))


if __name__ == '__main__':
    logger = logging.getLogger()
    logging.basicConfig(level=logging.DEBUG)
    if len(sys.argv) == 1:
        logger.info('Run for all N')
        pa_main()
    elif len(sys.argv) == 4:
        logger.info('Run for one instance')
        n = int(sys.argv[1])
        md = int(sys.argv[2])
        model = sys.argv[3]
        g_one_main(n, md, model)
