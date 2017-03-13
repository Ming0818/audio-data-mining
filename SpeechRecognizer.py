import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import numpy as np
import random
import math

mask_values = {}

# n is a key identifying the hash function, x is the value to be hashed
# references: http://stackoverflow.com/questions/2255604/hash-functions-family-generator-in-python
def hash_function(n, x):
    mask = mask_values.get(n)
    if mask is None:
        random.seed(n)
        mask = mask_values[n] = random.getrandbits(64)
    hashes = []
    m = pow(2, 14)
    for value in map(hash, x):
        hashes.append((value ^ mask) % m)

    return hashes


# t is the number of hash functions used to compute the min hash
def min_hash(t, set_one, set_two):

    min_hash_1 = []
    min_hash_2 = []

    for i in range(t):
        min_hash_1.append(math.inf)
        min_hash_2.append(math.inf)

    for i in range(t):
        hash1 = hash_function(i, set_one)
        min_hash_1[i] = min(hash1)

        hash2 = hash_function(i, set_two)
        min_hash_2[i] = min(hash2)

    js = 0

    for i in range(t):
        if min_hash_1[i] == min_hash_2[i]:
            js += 1

    return js / t


def k_gram(data, k):
    grams = set()

    end = len(data)
    i = k
    while i < end:
        gram = ''
        for j in range(k-1, 0, -1):
            gram += str(data[i-j][0])
            gram += str(data[i-j][1])
        gram = tuple(gram)
        if gram not in grams:
            grams.add(gram)
        i += 1

    return grams


file_base_m = './Audio_Samples/Subject_Meghan/Cat/cat'

mcat_kgrams = []

for i in range(10):
    file = file_base_m + str(i) + '.wav'
    rate, data = wav.read(file)
    data = data[:-1000]
    mcat_kgrams.append(k_gram(data, 3))

mcat_js = []

for i in range(1, len(mcat_kgrams)):
    mcat_js.append(min_hash(60, mcat_kgrams[0], mcat_kgrams[i]))

print(str(mcat_js))

file_base_g = './Audio_Samples/Subject_Gradey/Cat/cat'

gcat_kgrams = []

for i in range(10):
    file = file_base_g + str(i) + '.wav'
    rate, data = wav.read(file)
    data = data[:-1000]
    gcat_kgrams.append(k_gram(data, 3))

gcat_js = []

for i in range(1, len(gcat_kgrams)):
    gcat_js.append(min_hash(60, mcat_kgrams[0], gcat_kgrams[i]))

print(str(gcat_js))


#fft_out = np.abs(fft(data))

#plt.plot(data, fft_out)

# plt.plot(data)

# plt.show()