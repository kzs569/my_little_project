import itertools
from permutation import swap, reverse

# def _Combination(comb, length):
#     if comb is None:
#         return
#     if length == 0:
#         print()
#
#     _Combination(comb, length - 1)
#     _Combination(comb, length)


def Combination(comb):
    if comb is None:
        return

    result = []

    for i in range(1, len(comb) + 1):
        result.append(itertools.combinations(comb, i))

    for res in result:
        for i in res:
            print(i)

if __name__ == '__main__':
    str = 'abc'
    Combination(str)
