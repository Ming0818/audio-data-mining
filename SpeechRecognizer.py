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

# Tests for Cat
file_base_m = './Audio_Samples/Subject_Meghan/Cat/cat'

mcat_kgrams = []

for i in range(10):
    file = file_base_m + str(i) + '.wav'
    rate, data = wav.read(file)
    data = data[:-5000]
    new_data = []
    for dat in data:
        if np.abs(dat[0]) > 10 or np.abs(dat[1]) > 10:
            new_data.append(dat)
    mcat_kgrams.append(k_gram(new_data, 100))

mcat_js = []

for i in range(1, len(mcat_kgrams)):
    mcat_js.append(min_hash(60, mcat_kgrams[0], mcat_kgrams[i]))

print("Cat vs Cat")
for val in mcat_js:
    print('%.2f' % val)


# Tests for Hi
file_base_m = './Audio_Samples/Subject_Meghan/Hi/hi'

mhi_kgrams = []

for i in range(10):
    file = file_base_m + str(i) + '.wav'
    rate, data = wav.read(file)
    data = data[:-5000]
    new_data = []
    for dat in data:
        if np.abs(dat[0]) > 10 or np.abs(dat[1]) > 10:
            new_data.append(dat)
    mhi_kgrams.append(k_gram(new_data, 100))

mhi_js = []

for i in range(1, len(mhi_kgrams)):
    mhi_js.append(min_hash(60, mcat_kgrams[0], mhi_kgrams[i]))

print("Hi vs Cat")
for val in mhi_js:
    print('%.2f' % val)

# file_base_g = './Audio_Samples/Subject_Gradey/Cat/cat'
#
# gcat_kgrams = []
#
# for i in range(10):
#     file = file_base_g + str(i) + '.wav'
#     rate, data = wav.read(file)
#     data = data[:-5000]
#     new_data = []
#     for dat in data:
#         if np.abs(dat[0]) > 10 or np.abs(dat[1]) > 10:
#             new_data.append(dat)
#     gcat_kgrams.append(k_gram(new_data, 3))
#
# gcat_js = []
#
# for i in range(1, len(gcat_kgrams)):
#     gcat_js.append(min_hash(60, mcat_kgrams[0], gcat_kgrams[i]))
#
# print(str(gcat_js))



# file = './Audio_Samples/Subject_Meghan/Cat/cat0.wav'
# rate, raw_data = wav.read(file)
#
# data = raw_data[:-5000]
# new_data = []
# for dat in data:
#     if np.abs(dat[0]) > 10 or np.abs(dat[1]) > 10:
#         new_data.append(dat)
#
# # fft_out = np.abs(fft(data))
# #
# # plt.plot(data, fft_out)
#
# print(len(new_data))
# print(len(raw_data))
#
# plt.plot(raw_data)
#
# plt.plot(new_data)
#
# new_data = np.asarray(new_data)
#
# wav.write('cat_edit.wav', rate, new_data)
#
# plt.show()