import os
import numpy as np
from scipy.io import wavfile
import scipy.io

def find_wavs(dirpath):
    filelist = []
    for item in os.listdir(dirpath):
        if os.path.isfile(os.path.join(dirpath,item)) and item[-4:]=='.wav':
            filelist.append(os.path.join(dirpath,item))
        elif os.path.isdir(os.path.join(dirpath,item)):
            filelist.extend(find_wavs(os.path.join(dirpath,item)))
    return filelist

def make_mono(filelist, outdir):
    if not(os.path.exists(outdir)):
        os.makedirs(outdir)
    for file in filelist:
        fname = file.split(os.sep)[-1]
        fs, waveform = wavfile.read(file)
        wavfile.write(os.path.join(outdir,fname),fs,waveform[:, 0])
        '''
        if np.sum(np.abs(waveform[:, 0]-waveform[:, 1]))==0:
            rfiles=[]
            if fname[-5]=='i':
                i=1
                while os.path.exists(file[:-len(fname)]+fname[0]+'cod'+fname[4:-5]+str(i)+fname[-4:]):
                    rfiles.append(fname[0]+'cod'+fname[4:-5]+str(i)+fname[-4:])
            if os.path.exists(file[:-len(fname)]+fname[0]+'cod'+fname[4:]):
                rfiles.append(fname[0]+'cod'+fname[4:])
            for rfile in rfiles:
                rfs, rwaveform = wavfile.read(file[:-len(fname)]+rfile)
                if fs!=rfs:
                    raise Exception("reference and encoded fs not consistent!")
                wavfile.write(os.path.join(outdir,rfile),rfs,rwaveform[:, 0])
            
        else:
            wavfile.write(os.path.join(outdir,fname[0]+'err'+fname[4:]),fs,waveform[:, 0]-waveform[:, 1])
        '''


if __name__=="__main__":
    DIR = 'PEAQtestCD/Conformance'
    #DIR = 'PEAQtestCD/BS.1387-TestSet'
    OUTDIR = 'PEAQmono'

    flist = find_wavs(os.path.abspath(DIR))
    flist = make_mono(flist, os.path.abspath(OUTDIR))
