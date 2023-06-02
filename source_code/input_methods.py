import numpy as np


def input_validation(generators: np.array, signs: np.array) -> (np.array, np.array):
    work_gens = np.array([])
    work_signs = np.array([])

    if len(generators) != len(signs):
        print("Some generators might be missing their sign, or vice versa! Please check again...")
        return

    for g, s in zip(generators, signs):
        if int(s) != s:
            print("You have entered non integer signs for some generator! Please check again...")
            return
        else:
            s = int(s)
            work_gens = np.append(work_gens, np.tile(g, abs(s)))
            if s >= 0:
                work_signs = np.append(work_signs, np.tile(1, s))
            elif s < 0:
                work_signs = np.append(work_signs, np.tile(-1, abs(s)))

    return work_gens.astype(int), work_signs.astype(int)


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


# ===============================================================================
# ============================MAIN PART==========================================
gens = np.array([1, 2, 3, 4, 5])
sgns = np.array([1, -10, 5, -3, 0])

print(input_validation(gens, sgns))
print(write_word((gens, sgns)))
print(write_word(input_validation(gens, sgns)))
