import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import numpy as np

file_base = './Audio_Samples/Subject_Meghan/cat'

data_lists = []

for i in range(10):
    file = file_base + str(i) + '.wav'
    rate, data = wav.read(file)
    data_lists.append(data)

fft_out = fft(data_lists[0])

plt.plot(data, np.abs(fft_out))

plt.show()
