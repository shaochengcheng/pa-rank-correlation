from rank_correlation import g_one_main
import sys


if __name__ == '__main__':
    if len(sys.argv) == 4:
        n = int(sys.argv[1])
        md = int(sys.argv[2])
        model = sys.argv[3]
        g_one_main(2**n, md, model)
