import numpy as np
import torch
import torchaudio
try:
    import losslib.wfilters as wf
except:
    import wfilters as wf

class PreemLossParent(torch.nn.Module):
    def __init__(self, mode=0, N=2047):
        super().__init__()
        match mode:
            case 0:
                # a simple first order pre-emphasis (simple high pass)
                self.a=torch.tensor([1, 0])
                self.b=torch.tensor([1, -0.85])
            case 1:
                # folded differentiator
                self.a=torch.tensor([1, 0, 0])
                self.b=torch.tensor([1, 0, -0.85])
            case 2:
                # FIR approximation of A-curve plus a simple low pass
                # original paper used N=100
                self.a=torch.zeros(N+1)
                self.a[0]=1
                self.b = torch.from_numpy(np.convolve([1, 0.85], wf.Wcurve(N=N, mode=0))).type(torch.float)
            case 3:
                # outer and middle ear
                # approximation of the weighting function in ITU_T BS.1387 pg. 35
                self.a=torch.zeros(N)
                self.a[0]=1
                self.b=torch.from_numpy(wf.Wcurve(N=N, mode=1)).type(torch.float)
            case _:
                raise ValueError(f"Invalid loss type {mode}.")
    
    def preem(self, x):
        #x = torch.nn.functional.conv1d(x.unsqueeze(1), kernel, padding=1).squeeze(1)
        x=torchaudio.functional.filtfilt(x,self.a,self.b,clamp=True) # time must be the last dim of x
        return x

class MSeE(PreemLossParent):
    def __init__(self, mode=0, N=2047):
        super().__init__(mode, N)

    def forward(self, predictions, targets):
        predictions = self.preem(predictions)
        targets = self.preem(targets)
        
        return torch.mean((predictions - targets)**2)

class eESR(PreemLossParent):
    def __init__(self, mode=0, N=2047):
        super().__init__(mode, N)
    
    def forward(self,predictions,targets):
        predictions = self.preem(predictions)
        targets = self.preem(targets)

        return torch.sum(torch.abs(targets-predictions)**2)/torch.sum(torch.abs(targets)**2)

class eESR_DC(eESR):
    def __init__(self, mode=0, N=2047):
        super().__init__(mode, N)

    def forward(self, predictions, targets):
        DC = (torch.mean(targets-predictions)**2)/torch.mean(targets**2)
        ESR = super().forward(predictions,targets)
        return ESR+DC
    
class cd_lfcc(torch.nn.Module):
    def __init__(self, wlen=128, wstep=64, fs=44100, p=2.0):
        super().__init__()
        self.p=p
        self.wlen=wlen
        self.wstep=wstep
        self.fs=fs
        self.spec_tf=torchaudio.transforms.Spectrogram(n_fft=self.wlen,
                                                hop_length=self.wstep,
                                                window_fn=torch.hann_window,
                                                power=2,
                                                normalized=False,
                                                onesided=False)

    def forward(self, predictions, targets):
        predictions = self.cep(predictions)
        targets = self.cep(targets)
        distmat = torch.cdist(predictions, targets, p=self.p)
        return torch.mean(distmat, dim=(1,2))
    
    def cep(self, x):
        X=self.spec_tf(x)
        # returns a matrix with spectra in columns (wlen, wnum)
        Xl=torch.log(X)
        if len(Xl.shape) == 2:
            Xl = Xl.unsqueeze(0)
        cx = torch.real(torch.fft.ifft(Xl, n=None, dim=1, norm="backward"))
        cx = cx[:,0:(self.wlen//2)+1,:]
        return cx**2

class cd_mfcc(torch.nn.Module):
    def __init__(self, wlen=128, wstep=64, fs=44100, p=2.0):
        super().__init__()
        self.p=p
        self.wlen=wlen
        self.wstep=wstep
        self.fs=fs
        self.mfcc = torchaudio.transforms.MFCC(sample_rate=self.fs,
                                                  n_mfcc=40,
                                                  dct_type=2,
                                                  norm='ortho',
                                                  log_mels=False,
                                                  melkwargs={'n_fft':self.wlen,
                                                             'hop_length':self.wstep,
                                                             'n_mels':40,
                                                             'center':False})
    
    def forward(self, predictions, targets):
        predictions = self.mfcc(predictions)
        targets = self.mfcc(targets)
        distmat = torch.cdist(predictions, targets, p=self.p)
        return torch.mean(distmat, dim=(1,2))
    