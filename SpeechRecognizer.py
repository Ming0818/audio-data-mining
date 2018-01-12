import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
from scipy.fftpack import dct
import numpy as np
import random
import math
# from scipy.signal import

mask_values = {}

# n is a key identifying the hash function, x is the value to be hashed
# references: http://stackoverflow.com/questions/2255604/hash-functions-family-generator-in-python
def hash_function(n, x):
    mask = mask_values.get(n)
    if mask is None:
        random.seed(n)
        mask = mask_values[n] = random.getrandbits(64)
    hashes = []
    m = pow(2, 7)
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
    grams = []

    end = len(data)
    i = k
    while i < end:
        gram = []
        for j in range(k):
            for l in range(len(data[i-k+j])):
                gram.append(data[i-k+j][l])
        grams.append(tuple(gram))
        i += k

    return grams


def process_file(file):

    rate, raw_data = wav.read(file)

    raw_data = raw_data.T[0]

    data = raw_data[0:60000]

    # emphasize signal
    emphasis = 0.97
    data = np.append(data[0], data[1:] - emphasis * data[:-1])

    frame_size = 0.025  # 25 milliseconds
    frame_stride = 0.01  # 10 milliseconds
    frame_length = frame_size * rate  # in samples/frame
    frame_step = frame_stride * rate  # in samples/frame

    signal_length = len(data)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    # Make sure that we have at least 1 frame
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    # Pad Signal to make sure that all frames have equal number of samples
    # without truncating any samples from the original signal
    pad_signal = np.append(data, z)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    frames *= np.hamming(frame_length)

    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = (1.0 / NFFT) * (mag_frames ** 2)  # Power Spectrum

    nfilt = 40

    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # Keep 2-13

    (nframes, ncoeff) = mfcc.shape
    cep_lifter = 22
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift  # *

    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)

    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

    return [tuple(x.tolist()) for x in mfcc]

def process_file_raw(file):

    rate, raw_data = wav.read(file)

    raw_data = raw_data.T[0]

    data = raw_data[0:60000]

    # emphasize signal
    emphasis = 0.97
    data = np.append(data[0], data[1:] - emphasis * data[:-1])

    frame_size = 0.025  # 25 milliseconds
    frame_stride = 0.01  # 10 milliseconds
    frame_length = frame_size * rate  # in samples/frame
    frame_step = frame_stride * rate  # in samples/frame

    signal_length = len(data)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    # Make sure that we have at least 1 frame
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    # Pad Signal to make sure that all frames have equal number of samples
    # without truncating any samples from the original signal
    pad_signal = np.append(data, z)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    frames *= np.hamming(frame_length)

    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = (1.0 / NFFT) * (mag_frames ** 2)  # Power Spectrum

    return pow_frames


def create_wav(rate, data, name):

    data = np.asarray(data)

    wav.write(name, rate, data)

# Tests for Cat

file_base_m = './Audio_Samples/Subject_Meghan/Cat/cat'

data = process_file_raw(file_base_m + '0.wav')

freq = 44100

y_axis = range(len(data[0]))

plt.bar(y_axis, data[0], alpha=0.5)

label_points = [0, 64, 128, 192, 256]
inc = 44100 / 256
x_bars = [str(x * inc) for x in label_points]

plt.xticks(label_points, x_bars)

# plt.title("Single-Frame FFT")
# plt.xlabel("Frequency")
# plt.ylabel("Power")
# plt.show()

mcat_kgrams = []

total = 0
count = 0

for i in range(10):
    file = file_base_m + str(i) + '.wav'
    new_data = process_file(file)
    #mcat_kgrams.append(k_gram(new_data, 1))
    mcat_kgrams.append(new_data)

js_vals = []

for i in range(len(mcat_kgrams)):
    for j in range(i+1, len(mcat_kgrams)):
        js = min_hash(60, mcat_kgrams[i], mcat_kgrams[j])
        total += js
        count += 1
        js_vals.append(js)
        print('Cat' + str(i) + ' and Cat' + str(j) + ': ' + str(js))

avg_js = total / float(count)
js_vals = [(avg_js - val)**2 for val in js_vals]
variance = sum(js_vals) / len(js_vals)
print('Average JS: ' + str(total / count))
print('Standard Deviation: ' + str(math.sqrt(variance)))



file_base_m = './Audio_Samples/Subject_Meghan/Hi/hi'

mhi_kgrams = []

sum = 0
count = 0

for i in range(10):
    file = file_base_m + str(i) + '.wav'
    new_data = process_file(file)
    #mhi_kgrams.append(k_gram(new_data, 1))
    mhi_kgrams.append(new_data)

mhi_js = []

for i in range(len(mhi_kgrams)):
    for j in range(len(mcat_kgrams)):
        js = min_hash(60, mhi_kgrams[i], mcat_kgrams[j])
        sum += js
        count += 1
        print('Cat' + str(i) + ' and Hi' + str(j) + ': ' + str(js))

print('Average JS: ' + str(sum/count))








# file_base_m = './Audio_Samples/Subject_Meghan/Cat/cat'
#
# data = process_file(file_base_m + '0.wav')
#
# data = np.transpose(data)
#
# x_times = [math.floor((0.25 - .025) / .01), math.floor((0.5 - .025) / .01),
#            math.floor((0.75 - .025) / .01), math.floor((1 - .025) / 0.01), math.floor((1.25 - .025) / 0.01)]
# print(str(x_times))
# x_labels = ['0.25 sec', '0.5 sec', '0.75 sec', '1.0 sec', '1.25 sec']
#
# plt.imshow(data, cmap='hot', interpolation='nearest', aspect=8.0)
# plt.title("MFCC Heatmap (SF, 'cat')")
# plt.xlabel('Time')
# plt.ylabel('Mel-Frequency Cepstral Coefficients')
# plt.xticks(x_times, x_labels)
# plt.show()




# For the full MFCC
# mcat_kgrams = []
#
# total = 0
# count = 0
#
# for i in range(10):
#     file = file_base_m + str(i) + '.wav'
#     new_data = process_file(file)
#     # mcat_kgrams.append(k_gram(new_data, 3))
#     mcat_kgrams.append(new_data)
#
# js_vals = []
#
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
#
#
#
# file_base_m = './Audio_Samples/Subject_Meghan/Hi/hi'
#
# mhi_kgrams = []
#
# sum = 0
# count = 0
#
# for i in range(10):
#     file = file_base_m + str(i) + '.wav'
#     new_data = process_file(file)
#     #mhi_kgrams.append(k_gram(new_data, 3))
#     mhi_kgrams.append(new_data)
#
# mhi_js = []
#
# for i in range(len(mhi_kgrams)):
#     for j in range(len(mcat_kgrams)):
#         js = min_hash(60, mhi_kgrams[i], mcat_kgrams[j])
#         sum += js
#         count += 1
#         print('Cat' + str(i) + ' and Hi' + str(j) + ': ' + str(js))
#
# print('Average JS: ' + str(sum/count))
#
#
