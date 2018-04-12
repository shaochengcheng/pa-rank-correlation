from rank_correlation import g_one_main
import sys


def main_test(N, nrounds=1):
    n = 2**N
    for i in range(nrounds):
        for md in (1, 10):
            for model in ('pa', 'configuration'):
                g_one_main(n, md, model)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        main_test(int(sys.argv[1]))
    elif len(sys.argv) == 3:
        main_test(int(sys.argv[1]), int(sys.argv[2]))
    else:
        print('Only one or two parameters are allowed')
