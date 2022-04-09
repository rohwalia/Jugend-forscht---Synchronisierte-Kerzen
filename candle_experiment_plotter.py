import numpy as np
from scipy.integrate import odeint
import pandas as pd
from scipy import fftpack
import matplotlib.pyplot as plt
from scipy.signal import correlate

excel_file1 = '4_short.xlsx'
df1 = pd.read_excel(excel_file1, sheet_name='Series', dtpye=float, engine="openpyxl")
R_1 = df1['R'].tolist()
G_1 = df1['G'].tolist()
B_1 = df1['B'].tolist()
t_1 = df1['t'].tolist()

excel_file2 = 'radlum_2.xlsx'
df2 = pd.read_excel(excel_file2, sheet_name='Series', dtpye=float, engine="openpyxl")
R_2 = df2['R'].tolist()
G_2 = df2['G'].tolist()
B_2 = df2['B'].tolist()
t_2 = df2['t'].tolist()

R_1 = np.array(R_1)
G_1 = np.array(G_1)
B_1 = np.array(B_1)
R_2 = np.array(R_2)
G_2 = np.array(G_2)
B_2 = np.array(B_2)
t_1 = np.array(t_1)
t_2 = np.array(t_2)

dt = 1/240
up = 8
fft_R1 = fftpack.fft(R_1)
freqs = fftpack.fftfreq(R_1.size, d=dt)
power = np.abs(fft_R1)[np.where((freqs > up) & (freqs<15))]
freqs = freqs[np.where((freqs > up) & (freqs<15))]
freq_max = freqs[power.argmax()]
power_max = power[power.argmax()]
plt.plot(freqs, power)
plt.show()
print(freq_max)
print(power_max)

print(power.argsort()[-20:])
for i in power.argsort()[-20:]:
    print(freqs[i])
print("ok")

fft_R2 = fftpack.fft(R_2)
freqs = fftpack.fftfreq(R_2.size, d=dt)
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
    print(freqs[i])

lp_1 = np.sqrt(G_1**2 + B_1**2 + R_1**2)
lp_2 = np.sqrt(G_2**2 + B_2**2 + R_2**2)
plt.plot(t_1[:int(len(t_1)/2)], lp_1[:int(len(t_1)/2)])#, label = "Candle 1")
#plt.plot(t_2, lp_2, label= "Candle 2")
#plt.legend(loc=1)
plt.ylabel("Relative Helligkeit")
plt.xlabel("Zeit[s]")
plt.show()#plt.savefig("rad.svg")