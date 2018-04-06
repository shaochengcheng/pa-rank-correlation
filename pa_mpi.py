from rank_correlation import g_one_main
from mpi4py import MPI
import numpy as np


def chunk_seq(seq, n_chunk):
    out = []
    n = len(seq)
    chunk_size = n // n_chunk
    reminder = n % n_chunk
    first_right_edge = chunk_size + reminder
    out.append(seq[:first_right_edge])
    s = first_right_edge
    e = s + chunk_size
    while True:
        if s >= n:
            break
        out.append(seq[s:e])
        s = e
        e += chunk_size
    return out


def pa_mpi():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        N = np.logspace(19, 20, 2, base=2).astype(int)
        MD = [1, 10]
        MODELS = ['pa', 'configuration']
        data = [(n, md, model)
                for i in range(10) for n in N for md in MD for model in MODELS]
        data = chunk_seq(data, size)
    else:
        data = None
    sub_data = comm.scatter(data, root=0)
    for params in sub_data:
        # print('Process:%s is doing task of graph %s', rank, params)
        g_one_main(*params)


if __name__ == '__main__':
    pa_mpi()
