import matplotlib.pyplot as plt
from scipy.io import wavfile as wav

file_base_m = './Audio_Samples/Subject_Meghan/Cat/cat0.wav'

rate, data = wav.read(file_base_m)

x_values = range(len(data))

plt.plot(x_values, data)

plt.show()

