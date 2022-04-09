import numpy as np
from scipy.integrate import odeint
import pandas as pd
from scipy import fftpack
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import correlate

excel_file1 = 'rad_anti_1.xlsx'
df1 = pd.read_excel(excel_file1, sheet_name='Series', dtpye=float, engine="openpyxl")
h_1 = df1['h'].tolist()
t_1 = df1['t'].tolist()

excel_file2 = 'rad_anti_2.xlsx'
df2 = pd.read_excel(excel_file2, sheet_name='Series', dtpye=float, engine="openpyxl")
h_2 = df2['h'].tolist()
t_2 = df2['t'].tolist()

h_1 = np.array(h_1)
h_2 = np.array(h_2)
t_1 = np.array(t_1)
t_2 = np.array(t_2)

dt = 1/30
up = 1
fft_R1 = fftpack.fft(h_1)
freqs = fftpack.fftfreq(h_1.size, d=dt)
power = np.abs(fft_R1)[np.where((freqs > up) & (freqs<15))]
freqs = freqs[np.where((freqs > up) & (freqs<15))]
freq_max = freqs[power.argmax()]
power_max = power[power.argmax()]
"""plt.plot(freqs, power)
plt.ylabel("Nicht-normierte Amplitude")
plt.xlabel("Frequenz[Hz]")
plt.axvline(11.18756936736959, c="lightcoral", label="Frequenz: 11.18 Hz")
plt.legend()
plt.show()"""
print(freq_max)
print(power_max)

peak_max = signal.find_peaks(h_1[:int(len(t_1)/2)], 1100)
print(peak_max[0])
t_max = t_1[:int(len(t_1)/2)][peak_max[0]]
h_max = h_1[:int(len(t_1)/2)][peak_max[0]]

peak_min = signal.find_peaks(-h_1[:int(len(t_1)/2)], -950)
print(peak_min[0])
t_min = t_1[:int(len(t_1)/2)][peak_min[0]]
h_min = h_1[:int(len(t_1)/2)][peak_min[0]]

"""print(power.argsort()[-20:])
for i in power.argsort()[-20:]:
    print(freqs[i])
print("ok")

fft_R2 = fftpack.fft(h_2)
freqs = fftpack.fftfreq(h_2.size, d=dt)
power = np.abs(fft_R2)[np.where((freqs > up) & (freqs<15))]
freqs = freqs[np.where((freqs > up) & (freqs<15))]
freq_max = freqs[power.argmax()]
power_max = power[power.argmax()]
plt.plot(freqs, power)
plt.show()
print(freq_max)
print(power_max)

print(power.argsort()[-20:])
for i in power.argsort()[-20:]:
    print(freqs[i])"""

print(len(h_max))
print(len(h_min))

plt.plot(t_1, h_1, label = "Candle 1")
plt.plot(t_2, h_2, label= "Candle 2")
#plt.legend(loc=1)
#plt.scatter(t_max, h_max, c="gold", label="Maxima")
#plt.scatter(t_min, h_min, c="lightgreen", label="Minima")
plt.ylabel("Relative Höhe")
plt.xlabel("Zeit[s]")
plt.legend()
#plt.ylim(700,950)
#plt.xlim(6,7.1)
plt.show() #plt.savefig("height_canny.svg")"""

#Referenzlänge: 62.359 Pixel entsprechen 1.3 cm --> (1.3/62.359) cm pro Pixel
"""amplitude = np.average((h_max-h_min)/2)*(1.3/62.359)
amplitude_array = ((h_max-h_min)/2)*(1.3/62.359)
t_array = (t_max+t_min)/2
print(amplitude)
plt.plot(t_array, amplitude_array)
plt.axhline(amplitude, c="lightcoral", label="Amplitude: 2.69 cm")
plt.ylabel("Höhenamplitude[cm]")
plt.xlabel("Zeit[s]")
plt.legend()
plt.show() #plt.savefig("amplitude.svg")"""