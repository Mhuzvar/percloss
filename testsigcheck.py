import os
import numpy as np
from scipy.io import wavfile
import scipy.io

def load_and_check(filepath):
    fs, waveform = wavfile.read(filepath)
    if np.sum(np.abs(waveform[:][0]-waveform[:][1]))==0:
        print(filepath)


if __name__=="__main__":
    DIR = 'PEAQtestCD/Conformance/Data/Peaq'
    #DIR = 'PEAQtestCD/BS.1387-TestSet/'
    for file in os.listdir(DIR):
        load_and_check(os.path.join(DIR, file))