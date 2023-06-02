import numpy as np
from fractions import Fraction
from collections import Counter
# from math import gcd

# TODO: make the input format more user-friendly. Maybe read from a file or something...
"""
# Order of Braid group
n = int(input("Enter the order of the Braid group: "))
N = n**2 - n + 1
k = int(input("Enter the length of the word: "))

# word beta in Artin generators
generators = np.array([]).astype(int)
signs = np.array([]).astype(int)

# TODO: change input format. It's awful
for i in range(k):
    element = input(f"Generator Sign {i}: ").split()
    generators = np.append(generators, int(element[0]))
    signs = np.append(signs, int(element[1]))
"""


# ==========*************==============**************===============************============
# TODO: work on format of data: file or standard
# TODO: powers not only 1, -1
# TODO: powers not only 1, -1
# __________*************__________UTILITY FUNCTIONS___________****************______________
def write_word(word: tuple) -> str:
    """
    :param word: a tuple (gen, eps) with generators and epsilons
    :return: visual word with sigmas and superscripted epsilons
    """
    gen, eps = word
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    word = [("\N{GREEK SMALL LETTER SIGMA}" + str(gen[i])).translate(SUB)
            + (str(eps[i])).translate(SUP) for i in range(len(gen))]
    word = " ".join(word)
    return word


def delta(dim: int) -> tuple:
    """
    :param dim: order n of group B_n
    :return: Central element, i.e. Delta = s1s2...sn-1s1s2...sn-2....s1s2s3s1s2s1
    """
    d = np.array([])
    for j in range(1, dim):
        d = np.append(d, np.arange(1, dim + 1 - j))
    return d.astype(int), \
           np.ones(len(d)).astype(int)


def to_power(word: tuple, p: int) -> tuple:
    """
    :param word: a tuple (gen, eps) with the generators and their powers
    :param p: the power to be raised (can be negative)
    :return: the pair (new_gens, new_eps), i.e. the new word w^p
    """
    gens, epsilons = word
    if p >= 0:
        new_gens = np.tile(gens, p)
        new_epsilons = np.tile(epsilons, p)
    else:
        p *= -1
        new_gens = np.tile(np.flip(gens), p)
        new_epsilons = np.tile(np.flip(epsilons * (-1)), p)

    return new_gens, new_epsilons


def least_common(a: np.array) -> int:
    """
    returns the least common element in an array
    """
    a = Counter(a)
    return a.most_common()[-1][0]


def mult_words(word1: tuple, word2: tuple) -> tuple:
    """
    :param word1: tuple (gen1, eps1) for the word w1
    :param word2: tuple (gen2, eps2) for the word w2
    :return: the product word w1w2, i.e. the concatenation of both words (no reductions)
    """
    gens1, eps1 = word1
    gens2, eps2 = word2

    new_gens = np.hstack((gens1, gens2))
    new_eps = np.hstack((eps1, eps2))
    return new_gens, new_eps


def is_between(x, lb, ub):
    return lb <= x <= ub


# __________*************__________COMPARISON FUNCTIONS___________****************______________
def compare_with_delta(n: int, word: tuple, m: int, details: bool) -> int:
    """
    :param n: index of Group B_n
    :param word: tuple (gen, eps) with the word beta
    :param m: power of delta word to compare with
    :param details: indicator to print the result of every step or not
    :return: We want to compare Delta^m * beta^{-1} with the unit word
             -1 -- if Delta^m < beta
              0 -- if Delta^m = beta
              1 -- if Delta^m > beta
    """
    # Needed for procedure below
    # NO NUMPY HERE, otherwise there are troubles with large integers
    A = [0] + [cut for cut in range(n - 1, -1, -1)] + [0]
    B = [0] + [1 for _ in range(n)]
    C = [0] + [cut for cut in range(n - 1, -1, -1)] + [0]

    # A = np.hstack((0, np.arange(n - 1, -1, -1), 0))
    # B = np.hstack((0, np.ones(n))).astype(int)

    # A = np.array(A, dtype='int64')
    # B = np.array(B, dtype='int64')
    # C = np.array(C, dtype='int64')

    # new_word = Delta^m * beta^{-1}
    new_word = mult_words(to_power(delta(n), m), to_power(word, -1))
    if details:
        print("**************************************")
        print(f"Braid word to act on the diagram: {write_word(new_word)}")
        print("Initial D:        ", A, B, C)

    new_gens = new_word[0]
    new_epsilons = new_word[1]

    def psi(i: int, eps: int) -> None:
        """
        :param i: action by i-th generator on diagram D (i.e. arrays A, B, C)
        :param eps: the corresponding power of the generator
        :return: None; it changes the values of A, B, C accordingly to the action
        """
        if eps == 1:
            b1 = max(B[i - 1] + C[i + 1], B[i] + C[i - 1]) - C[i]
            b2 = max(A[i + 2] + B[i], A[i] + B[i + 1]) - A[i + 1]
            a1 = max(A[i - 1] + B[i], A[i] + b1) - B[i - 1]
            c1 = max(C[i + 2] + B[i], C[i + 1] + b2) - B[i + 1]
            A[i + 1] = A[i]
            A[i] = a1
            B[i - 1] = b1
            B[i + 1] = b2
            C[i] = C[i + 1]
            C[i + 1] = c1
        elif eps == -1:
            b1 = max(A[i - 1] + B[i], A[i + 1] + B[i - 1]) - A[i]
            b2 = max(B[i] + C[i + 2], B[i + 1] + C[i]) - C[i + 1]
            a1 = max(B[i] + A[i + 2], A[i + 1] + b2) - B[i + 1]
            c1 = max(B[i] + C[i - 1], C[i] + b1) - B[i - 1]
            A[i] = A[i + 1]
            A[i + 1] = a1
            B[i - 1] = b1
            B[i + 1] = b2
            C[i + 1] = C[i]
            C[i] = c1
        else:
            print("You have entered a non-valid signs for generators! They should be +- 1. "
                  "If some generator is in some power, write it separately as a product with itself.")

    def theta(x: list) -> int:
        """
        :param x: array A - C, where A, C are defined above
        :return: -1 -- if A - C < 0
                  0 -- if A - C = 0
                  1 -- if A - C > 0
        """
        for v in x:
            if v != 0:
                return v // abs(v)  # if negative, Delta*m < beta, otherwise, Delta^m > beta
        return 0  # here is equal

    def apply_gens(gen: np.array) -> None:
        """
        :param gen: generators of the word that is to be compared with the unit word
        :return: None; applies psi procedure
        """
        for ij, w in enumerate(gen):
            psi(w, new_epsilons[ij])
            if details:
                print(f"D after {ij + 1}-th generator: ", A, B, C)

    # Compare Delta^m * beta^-1 with e
    apply_gens(new_gens)
    AC = np.subtract(A, C)
    # print(theta(AC))
    return theta(AC)


# __________*************__________CALCULATING FDTC___________****************______________
def search_bounds(n: int, word: tuple) -> tuple:
    """
    calculates the number of the least common generator
    under negative power (left) and positive power (right)
    """
    gen, eps = word
    word_help = gen * eps

    neg_word = word_help[word_help < 0] * (-1)
    l_least = least_common(np.hstack((neg_word, np.arange(1, n))))
    left = -np.count_nonzero(neg_word == l_least)  # note that we say left = -left

    pos_word = word_help[word_help > 0]
    r_least = least_common(np.hstack((pos_word, np.arange(1, n))))
    right = np.count_nonzero(pos_word == r_least)

    left -= 1  # since we search the floor F, and w - 1 <= F <= w
    # on the other hand, left <= w <= right
    # so, left - 1 <= F <= right

    return left, right


def dehornoy_floor(n: int, word: tuple, details: bool) -> int:
    """
    :param n: index of Group B_n
    :param word: tuple (gen, eps): the beta word, whose floor we calculate
    :param details: indicator to print the result of every step or not
    :return: Dehornoy floor, i.e. the value z, s.t. Delta^2z <= beta < Delta^2(z+1)
    """
    N = n**2 - n + 1
    left, right = search_bounds(n, word)
    if details:
        print("Left, right bounds for search are [" + str(left) + ", " + str(right) + "]")
        print("Entering dehornoy floor calculation...")
        print("======================================")
    ph = 0
    while left <= right:
        mid = left + (right - left) // 2
        if details:
            if ph != 0:
                print(f"Phase {ph} of the search: repeating search with z = {mid}")
            else:
                print(f"Phase 0 of the search: trying for z = {mid}")

        if compare_with_delta(n, word, 2 * mid, details) == 1:
            if details:
                print(f"During phase {ph} of the search, we got that beta < Delta^{2 * mid}")
            right = mid - 1
        else:
            if details:
                print(f"*=*=*=*=*=  Beta is indeed >= Delta^{2 * mid} *=*=*=*=*=")
            if compare_with_delta(n, word, 2 * (mid + 1), details) != 1:
                if details:
                    print(f"During phase {ph} of the search, we got that beta > Delta^{2 * (mid + 1)}")
                left = mid + 1
            else:
                if details:
                    print(f"*=*=*=*=*=  Beta is indeed < Delta^{2 * (mid + 1)} *=*=*=*=*=")
                    print(f"Hence, during phase {ph} of the search, we got that "
                          f"Delta^{2 * mid} <= beta < Delta^{2 * (mid + 1)}")
                    print(f"Dehornoy floor of beta^{N} is", mid)
                return mid

        ph += 1
        if details:
            print("==================================")

    return 0


def calculate_twist(n: int, word: tuple, details=False) -> Fraction:
    """
    :param n: index of Group B_n
    :param word: tuple (gen, eps): the beta word, whose twist we calculate
    :param details: indicator to print the result of every step or not
    :return: FDTC of word
    """
    N = n ** 2 - n + 1
    z = dehornoy_floor(n, to_power(word, N), details)

    if details:
        print("==================================")

    low_bound = Fraction(z, N)
    up_bound = Fraction(z + 1, N)
    if details:
        print("Lower bound for FDTC z/N =", low_bound)
        print("Upper bound for FDTC (z+1)/N =", up_bound)
        print(f"Searching for the unique fraction with denominator <= {n} in [{low_bound}, {up_bound}]...")

    frac = low_bound.limit_denominator(n)
    if is_between(frac, low_bound, up_bound):
        return frac

    frac = up_bound.limit_denominator(n)
    if is_between(frac, low_bound, up_bound):
        return frac

    for c in range(1, n + 1):
        for d in range(z * c // N, (z + 1) * c // N + 1):
            frac = Fraction(d, c)
            if is_between(frac, low_bound, up_bound):
                return frac

    frac = Fraction(0)
    return frac


# # _______********** SAVE SOME DATA TO CHECK IF STH IS WRONG
# generators = braid[0]
# signs = braid[1]
# # df = dehornoy_floor(braid)
# f = open(f"test{generators[:10]}.txt", "w+")
# f.write(str(ind))
# f.write('\n')
# f.write(str(generators))
# f.write('\n')
# f.write(str(signs))
# # f.write('\n')
# # f.write(str(df) + "  " + str(Fraction(df, N)))
# f.write('\n')
# f.write(str(fdtc))
# # f.write('\n')
# # f.write(str(df/N < fdtc < (df+1)/N))
# f.close()
