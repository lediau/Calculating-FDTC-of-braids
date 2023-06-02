from fdtc_functions import *
import re


def pair_format() -> tuple:
    generators = np.array([]).astype(int)
    signs = np.array([]).astype(int)

    k = int(input("Enter the length of the word: "))
    print("Now enter in each row the generator index and its sign, space separated")
    print("Indexing starts from 1 and ends with n-1")
    print("For example, you can enter in a row 3 -2 meaning that it is sigma3 in power -2")

    for i in range(k):
        element = input(f"{i}th Generator Sign: ").split()
        generators = np.append(generators, int(element[0]))
        signs = np.append(signs, int(element[1]))

    return generators, signs


def array_format() -> tuple:
    generators = np.array([]).astype(int)
    signs = np.array([]).astype(int)

    print("Now enter in the same row the generators in the order they appear in the word")
    print("Indexing starts from 1 and ends with n-1")
    print("For example, you can enter 1 2 3 4 5 meaning that your generators are sigma1, sigma2, etc...")
    gen = input("Generators: ").split()
    gen = [int(g) for g in gen]
    generators = np.append(generators, gen)

    print("Now enter in the same row the signs of the generators provided above")
    print("For example, you can enter -2 5 1 4 -6 meaning that your word is sigma1^{-2}sigma2^{5}...")
    sgns = input("Signs: ").split()
    sgns = [int(s) for s in sgns]
    signs = np.append(signs, sgns)

    return generators, signs


def file_method(fn):
    with open(f'{fn}.txt') as fp:
        data = [tuple(np.fromiter(map(int, line.strip().split()), dtype=int)) for line in fp]
        return data


print("*=*=*=*=*=*=Program to calculate FDTC of a given braid*=*=*=*=*=*=")
# Choosing the input method
while True:
    input_method = input("Choose the number input method you would prefer (1, 2, or 3):\n\
    1. Standard input in pair format: generator sign, generator sign, ..., generator sign in every row\n\
    2. Standard input in array format: generators as a separate array, and the same for signs\n\
    3. File format: put a text file that fits the description in the same directory\n")

    if re.match('[1-3]', input_method):
        input_method = int(input_method[0])
        break
    else:
        print("You have to input a number, namely 1, 2, or 3...")

# Input the braid group index
while True:
    n = input("Enter the order of the Braid group: ")
    if n.isnumeric():
        n = int(n)
        if n > 0:
            break
        else:
            print("You have to input a natural number, namely the index n of the braid group Bn")
    else:
        print("You have to input a natural number, namely the index n of the braid group Bn")

# ALL CASES:
word = (np.array([]), np.array([]))
# 1. If it is pair method
if input_method == 1:
    word = pair_format()

# 2. If it is array method
elif input_method == 2:
    word = array_format()

# 3. If it is file method
elif input_method == 3:
    print("Enter the name of your file (no need to include .txt)")
    print("So, if your file is named my_file.txt, input only my_file")
    file_name = input()
    word = file_method(file_name)

# Print the braid
print("You have entered the following braid:")
written_word = write_word(word)
print(written_word)
print(f"We are working in the group B_{n}:")

# PREPROCESS THE WORD
word = input_validation(word)

# STARTING calculations
print("Now we are ready to calculate the FDTC of the braid...")
yn = input("(y/n)Write y if you want the calculation details displayed, and n otherwise: ")
if yn.lower() == 'y':
    detailed = True
else:
    detailed = False

print(f"""The FDTC of the word {written_word}
                   in the group B_{n}
                   is equal to""", calculate_twist(n, word, detailed), ".")

# ---========--------==============------------=================-----------------================
# TODO:===========YOU CAN MANUALLY CHANGE THE PART BELOW AND COMMENT WHAT IS ABOVE===============
# n = 6
# generators = np.array([1, 2, 3, 4, 5])
# signs = np.array([1, -2, 1, -9, 8])
#
# word = (generators, signs)
#
# word = input_validation(word)
#
# calculate_twist(n, word, True)
