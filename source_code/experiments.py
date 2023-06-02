from main import *
import numpy as np
# from fractions import Fraction
import pandas as pd
import time


def rand_word(ind: int, length: int) -> tuple:
    gens = np.random.choice(np.arange(1, ind), length)
    signs = np.random.choice(np.array([-1, 1]), length)
    word = (gens, signs)
    return word


def uniform_symbol(ind: int) -> (int, int):
    gen = np.random.choice(np.arange(1, ind), 1)
    sign = np.random.choice(np.array([-1, 1]), 1)
    return gen, sign


def positive_uniform_symbol(ind: int) -> (int, int):
    gen = np.random.choice(np.arange(1, ind), 1)
    sign = 1
    return gen, sign


def nonsymmetric_symbol(ind: int) -> (int, int):
    gen = np.random.choice(np.arange(1, ind), 1, p=[0.6, 0.3, 0.1])
    sign = np.random.choice(np.array([-1, 1]), 1)
    return gen, sign


def nonuniform_symbol(ind: int) -> (int, int):
    gen = np.random.choice(np.arange(1, ind), 1, p=[0.6, 0.3, 0.1])
    sign = np.random.choice(np.array([-1, 1]), 1, p=[0.25, 0.75])
    return gen, sign


def random_walk(n: int, k: int, s: int) -> list:
    # to store all the words
    words = []

    # generate a base word beta_0
    braid = rand_word(n, k)
    words.append(braid)

    # generate words using random walk, adding one symbol per turn
    for _ in range(s):
        next_symbol = positive_uniform_symbol(n)
        braid = (np.append(braid[0], next_symbol[0]),
                 np.append(braid[1], next_symbol[1]))
        words.append(braid)

    return words


def results(n: int, words: list, distribution="Uniform") -> list:
    all_frac_twists = []
    # we have all the words of the chain
    for w in words:
        print(write_word(w))
        print(w, file=f)

    # calculating their twists and their fractional parts
    for i, w in enumerate(words):
        twist = calculate_twist(n, w)
        print(f"{i}:", twist)
        print(f"FDTC of word {i}:", twist, file=f)
        twist = twist - twist.numerator // twist.denominator
        all_frac_twists.append(twist)

    print(all_frac_twists)

    # printing the frequency table
    print("====FREQUENCY TABLE======")
    print(f"For {distribution} distribution, in B_{n} we have:")
    print("====FREQUENCY TABLE======", file=f)
    print(f"For {distribution} distribution, in B_{n} we have:", file=f)

    col1 = pd.Series(all_frac_twists, name="Fractional parts")
    df = col1.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
    print(df)
    print(df, file=f)
    df.to_csv("results.csv", encoding='utf-8', index=True)
    return all_frac_twists


def trans_results(n: int, f_twists: list, distribution="Uniform") -> list:
    transition_twists = [(f_twists[i], f_twists[i + 1]) for i in range(len(f_twists) - 1)]
    print(transition_twists)
    # printing the frequency table
    print("====TRANSITION FREQUENCY TABLE======")
    print(f"Transitions for {distribution} distribution, in B_{n} are:")
    print("====TRANSITION FREQUENCY TABLE======", file=f)
    print(f"For {distribution} distribution, in B_{n} we have:", file=f)

    col1 = pd.Series(transition_twists, name="Transitions")
    df = col1.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
    print(df)
    print(df, file=f)
    df.to_csv("trans_results.csv", encoding='utf-8', index=True)
    return transition_twists


# =====================***************MAIN PART***************=======================
start_time = time.time()

# fix all parameters
index = 4  # braid group index B_n
base_length = 1  # length of beta_0, the base word
chain_length = 100  # length of the chain

f = open(f"exp3_{index}_1000_pos_uniform.txt", "a")  # file to store results

all_words = random_walk(index, base_length, chain_length)

frac_twists = results(index, all_words)

transition_pairs = trans_results(index, frac_twists)


# ==============***************END OF FIRST EXPERIMENT****************===============

print("--- %s seconds ---" % (time.time() - start_time))

