import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torchaudio
import numpy as np
import scipy

import losslib.percloss as pl
import losslib.wfilters as wf

fs = 44100

wA=np.linspace(0,22100,512)
Ra = ((12194**2)*(wA**4))/(((wA**2)+(20.6**2))*np.sqrt(((wA**2)+(107.7**2))*((wA**2)+(737.9**2)))*((wA**2)+(12194**2)))
hA=20*np.log10(Ra)+2

title=['HP',
       'Folded Differentiator',
       '99-point A-weight approximation + LP',
       '63-point BS.468 approximation']
for i in range(4):
    if i==2:
        N=99
    else:
        N=63
    msee = pl.MSeE(mode=i, N=N)
    if i == 3:
        w, h = scipy.signal.freqz(msee.b, a=msee.a, fs=fs)
    else:
        w, h = scipy.signal.freqz(msee.b, fs=fs)
    plt.figure(i)
    if i >1:
        if i==3:
            wA = (31.5, 63, 100, 200, 400, 800, 1000, 2000, 3150, 4000, 5000,
                    6300, 7100, 8000, 9000, 10000, 12500, 14000, 16000, 20000, 31500)
            hA = (-29.9, -23.9, -19.8, -13.8, -7.8, -1.9, 0, 5.6, 9, 10.5, 11.7,
                    12.2, 12, 11.4, 10.1, 8.1, 0, -5.3, -11.7, -22.2, -42.7)
            plt.semilogx(wA,hA,color='tab:orange')
        else:
            wlp, hlp = scipy.signal.freqz([1, 0.85], fs=fs)
            plt.semilogx(wA,hA+20*np.log10(hlp),color='tab:orange')
        
    plt.semilogx(w,20*np.log10(np.abs(h)), color='tab:blue')
    plt.grid()
    plt.axis([2e1, 2e4, -20, 15])
    plt.title(title[i])
    plt.xticks([2e1, 2e2, 2e3, 2e4], [20, 200, '2k', '20k'])
plt.show()
