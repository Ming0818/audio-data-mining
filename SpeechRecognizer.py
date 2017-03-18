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
        gram = tuple(data[i-k:i])
        grams.add(gram)
        i += 1

    return grams


def process_file(file):

    rate, raw_data = wav.read(file)

    raw_data = raw_data.T[0]

    data = raw_data[:-5000]
    new_data = []
    for dat in data:
        if np.abs(dat) > 10:
            new_data.append(dat)

    max_val = float(max(new_data))
    new_data = [((val/max_val)*2)-1 for val in new_data] # Normalize the data

    return new_data


def create_wav(rate, data, name):

    data = np.asarray(data)

    wav.write(name, rate, data)

# Tests for Cat

file_base_m = './Audio_Samples/Subject_Meghan/Cat/cat'

mcat_kgrams = []

total = 0
count = 0

for i in range(10):
    file = file_base_m + str(i) + '.wav'
    new_data = process_file(file)
    mcat_kgrams.append(k_gram(new_data, 10))

js_vals = []

# for i in range(len(mcat_kgrams)):
#     for j in range(i+1, len(mcat_kgrams)):
#         js = min_hash(60, mcat_kgrams[i], mcat_kgrams[j])
#         total += js
#         count += 1
#         js_vals.append(js)
#         print('Cat' + str(i) + ' and Cat' + str(j) + ': ' + str(js))
#
# avg_js = total / float(count)
# js_vals = [(avg_js - val)**2 for val in js_vals]
# variance = sum(js_vals) / len(js_vals)
# print('Average JS: ' + str(total / count))
# print('Standard Deviation: ' + str(math.sqrt(variance)))



file_base_m = './Audio_Samples/Subject_Gradey/Cat/cat'

mhi_kgrams = []

sum = 0
count = 0

for i in range(10):
    file = file_base_m + str(i) + '.wav'
    new_data = process_file(file)
    mhi_kgrams.append(k_gram(new_data, 10))

mhi_js = []

for i in range(len(mhi_kgrams)):
    for j in range(len(mcat_kgrams)):
        js = min_hash(60, mhi_kgrams[i], mcat_kgrams[j])
        sum += js
        count += 1
        print('Cat' + str(i) + ' and Hi' + str(j) + ': ' + str(js))

print('Average JS: ' + str(sum/count))




# data = process_file('./Audio_Samples/Subject_Meghan/Cat/cat0.wav')
#
# fft_out = np.abs(fft(data))
#
# print(fft_out[0])

# plt.plot(data, fft_out)
#
# plt.show()