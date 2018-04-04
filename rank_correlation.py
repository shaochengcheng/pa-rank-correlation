import graph_tool.all as gt
import numpy as np
import pandas as pd
from scipy import stats
from multiprocessing import Process, Manager
from queue import Empty
import pickle

import logging
import sys

logger = logging.getLogger(__name__)


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


def g_stats(g):
    return dict(
        n=g.num_vertices(),
        m=g.num_edges(),
        l=average_path(g),
        r_a=gt.assortativity(g, 'total'),
        rho_a=rank_assortativity(g, method='spearman'),
        tau_a=rank_assortativity(g, method='kendall'),
        cc=gt.global_clustering(g),)


def g_centrality_correlations(g):
    dgr = g.degree_property_map('total').a
    dgr = dgr / (g.num_vertices() - 1)
    btn = gt.betweenness(g, norm=True)[0].a
    cln = gt.closeness(g, norm=True, harmonic=False).a
    egn = gt.eigenvector(g)[1].a
    return dict(
        dgr=dgr,
        btn=btn,
        cln=cln,
        egn=egn,
        db_p=stats.pearsonr(dgr, btn),
        dc_p=stats.pearsonr(dgr, cln),
        de_p=stats.pearsonr(dgr, egn),
        db_s=stats.spearmanr(dgr, btn),
        dc_s=stats.spearmanr(dgr, cln),
        de_s=stats.spearmanr(dgr, egn),
        db_k=stats.kendalltau(dgr, btn),
        dc_k=stats.kendalltau(dgr, cln),
        de_k=stats.kendalltau(dgr, egn))


def producer_queue(q1, N, M_delta):
    for n in N:
        for m_delta in M_delta:
            q1.put((n, m_delta, 'pa'))
            q1.put((n, m_delta, 'configuration'))
    logger.info('All parameters are put into q1.')
    logger.info('N=%s', N)
    logger.info('M_delta=%s', M_delta)


def workers_queue(pid, q1, q2):
    while True:
        try:
            data = q1.get(timeout=1)
        except Empty:
            logger.info('Work process %s idles for 1 second, stop it!', pid)
            q2.put((pid, None, 'STOP'))
            break
        if data == 'STOP':
            logger.info('Work process %i received STOP!', pid)
            q1.put('STOP')
            q2.put((pid, None, 'STOP'))
            break
        n, m_delta, model = data
        g = gt.price_network(
            N=n,
            m=m_delta,
            gamma=1,
            seed_graph=gt.complete_graph(N=m_delta))
        if model == 'configuration':
            gt.random_rewire(g, model='configuration')
        try:
            r1 = g_stats(g)
            r = g_centrality_correlations(g)
            r['n'] = n
            r['m_delta'] = m_delta
            r['model'] = model
            for k, v in r1.items():
                r[k] = k
        except Exception as e:
            logger.error(e)
            r = dict(n=n, m_delta=m_delta, model=model)
        q2.put((pid, r, 'RUN'))


def collector_queue(q2, number_of_workers, filename):
    rs = []
    workers_status = [1 for i in range(number_of_workers)]
    while True:
        pid, r, status = q2.get()
        if status == 'STOP':
            logger.info(
                'Collector process: STOP sign of worker process %s received from q2',
                pid)
            workers_status[pid] = 0
            if sum(workers_status) == 0:
                logger.warning('All STOP signs received from q2.')
                pickle.dump(rs, filename, -1)
                logger.warning('Results collected and saved!')
                break
        else:
            logger.info('Collector process: receiving result from %s', pid)
            rs.append(r)


class PaManager(object):

    def __init__(self, number_of_workers, filename, N, M_delta):
        self.manager = Manager()
        self.q1 = self.manager.Queue()
        self.q2 = self.manager.Queue()
        self.number_of_workers = number_of_workers
        self.filename = filename
        self.N = N
        self.M_delta = M_delta

    def start(self):
        self.producer = Process(
            target=producer_queue, args=(self.q1, self.N, self.M_delta))
        self.producer.start()

        self.workers = [
            Process(target=workers_queue, args=(i, self.q1, self.q2))
            for i in range(self.number_of_workers)
        ]
        for workers in self.workers:
            workers.start()

        self.collector = Process(
            target=collector_queue,
            args=(self.q2, self.number_of_workers, self.filename))
        self.collector.start()

    def join(self):
        self.producer.join()
        for workers in self.workers:
            workers.join()
        self.collector.join()


def mp_pa_main(number_of_workers=4,
               filename='pa.pkl',
               N=np.logspace(10, 16, 6, base=2).astype(int),
               M_delta=[1, 10]):
    try:
        manager = PaManager(
            number_of_workers=number_of_workers,
            filename=filename,
            N=N,
            M_delta=M_delta)
        manager.start()
        manager.join()
    except (KeyboardInterrupt, SystemExit):
        logger.info('interrupt signal received')
        sys.exit(1)
    except Exception as e:
        raise e


if __name__ == '__main__':
    logger = logging.getLogger()
    logging.basicConfig(level=logging.DEBUG)
    mp_pa_main()
