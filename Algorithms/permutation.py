def swap(perm, begin, end):
    strs = []
    for i in range(len(perm)):
        if i == begin:
            strs.append(perm[end])
        elif i == end:
            strs.append(perm[begin])
        else:
            strs.append(perm[i])
    # print(strs)
    return ''.join(strs)


def reverse(perm, begin, end):
    rev = [c for c in perm[begin: end + 1][::-1]]
    strs = []
    for i in range(len(perm)):
        if i >= begin and i <= end:
            strs.append(rev.pop(0))
        else:
            strs.append(perm[i])
    # print(strs)
    return ''.join(strs)


# 字符串非重复全排列(递归实现)
def Permutation(perm, begin, end):
    if end < 1:
        return
    if begin == end:
        print(perm)
    else:
        for i in range(begin, end + 1):
            perm = swap(perm, i, begin)
            # print(begin,end)
            Permutation(perm, begin + 1, end)
            perm = swap(perm, i, begin)


# 字符串非重复全排列（字典序）
def next_permutation(perm):
    # 找到排列中最右一个升序的首位位置i
    i = len(perm) - 2
    while i >= 0 and perm[i] >= perm[i + 1]:
        i -= 1
    if i < 0:
        return None
    # 找到排列中第i位右边最后一个比perm[i]大的位置k
    k = len(perm) - 1
    while k > i and perm[k] <= perm[i]:
        k -= 1
    # 交换k，i位置的元素
    perm = swap(perm, k, i)
    # 翻转i右边的元素
    perm = reverse(perm, i + 1, len(perm))
    return perm


def Permutation_dic(perm):
    strs = ''.join(sorted([c for c in perm]))  # 字符串有序
    print('initial state:', strs)
    while strs is not None:
        strs = next_permutation(strs)
        print(strs)

if __name__ == '__main__':
    str = 'bca'
    Permutation_dic(str)


#itertools.permutations(,)了解下