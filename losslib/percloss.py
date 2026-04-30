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

class PEAQ(torch.nn.Module):
    def __init__(self, fs=44100):
        super().__init__()
        if fs<2040:
            raise Exception(f"Cannot calculate normalization factor for fs = {fs} Hz!")
        elif fs<36000:
            print("\x1b[0;37;43mWarning:\x1b[0m fs lower than 36 kHz may introduce problems during calculation!")
        self.fs = fs
        self.nfft = int(np.floor(2048*(self.fs/48000)))
        self.step = int(np.ceil(self.nfft/2))
        self.get_normfac()
        self.barkmat, self.imin, self.imax = wf.lin2bark_mat(self.fs, self.nfft)
        
    def get_normfac(self):
        # calculate from 10 frames of 0 dB full scale 1019.5 Hz sine wave
        f = 1019.5
        slen = 9*self.step+self.nfft
        t = torch.arange(slen)/self.fs
        swave = torch.sin(2*torch.pi*f*t)
        normfac = 0
        for i in range(10):
            win = torch.hann_window(self.nfft)*swave[i*self.step:i*self.step+self.nfft]
            normfac += torch.max(torch.abs(torch.fft.rfft(win)))/10
        Lp = 92                         # default value
        self.fac = (10**(Lp/20))/normfac
    
    def forward(self, predictions, targets):
        if len(predictions.shape) == 1:
            self.bsize = 1
        else:
            self.bsize = predictions.shape[0]
        #MOVs = torch.empty((self.bsize, 11), requires_grad=True)
        MOVs = 11*[0]

        ep, mp, modEl, sp, noisep, MOVs[5], Eth = self.pem(predictions, targets)
            # reference is always the latter half in batch
            # MOV index 5 is EHS_B (p. 68)
        
        # masker
        msk = self.calc_mask(ep[self.bsize:,:,:])

        slp = self.pat_adap_LP(ep)
            # low passed excitation patterns
        t_slp, r_slp = self.lev_adap(slp)
            # level adaptation
        t_Ep, r_Ep = self.pat_adap(t_slp, r_slp)
            # spectrally adapted patterns E_{P,x}

        # specific loudness patterns
        #N, Eth = self.calc_loud(ep)
            # Neither of these is ever used in the MOVs calculation

        # calculation of remaining MOVs
        MOVs[:5], MOVs[6:] = self.calc_MOV(mp, modEl[self.bsize:,:,:], Eth, t_Ep, r_Ep, 20*torch.log10(torch.abs(sp)), noisep, msk.transpose(1,2), ep)
        MOVs = torch.stack(MOVs, dim=-1)
        
        amin = torch.tensor([393.915565, 361.965332, -24.046116,   1.110661, -0.206623,  0.074318,  1.113683,    0.950345,  0.029985, 0.000101, 0])
        amax = torch.tensor([921       , 881.131226,  16.212030, 107.137772,  2.886017, 13.933351, 63.257874, 1145.018555, 14.819740, 1       , 1])
        
        x = (MOVs-amin)/(amax-amin)
        nodes = torch.matmul(x,torch.tensor([[-0.502657,  0.436333,  1.219602],[ 4.307481,  3.246017,  1.123743],[ 4.984241, -2.211189, -0.192096],[ 0.051056, -1.762424,  4.331315],[ 2.321580,  1.789971, -0.754560],[-5.303901, -3.452257, -10.814982],[ 2.730991, -6.111805,  1.519223],[ 0.624950, -1.331523, -5.955151],[ 3.102889,  0.871260, -5.922878],[-1.051468, -0.939882, -0.142913],[-1.804679, -0.503610, -0.620456]], dtype=torch.double))
        nodes = nodes+torch.tensor([-2.518254,0.654841,-2.207228])
        nodes = 1/(1+torch.exp(torch.clamp(-nodes, max=7.0978e2)))
        DI = torch.matmul(nodes,torch.tensor([-3.817048,4.107138,4.629582], dtype=torch.double))-0.307594
        ODG = -3.98+4.2/(1+torch.exp(-DI))
        return torch.mean(ODG)

    def pem(self, x, y):

        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        x_cat = torch.cat((x,y), dim=0)
            # makes both x and y be considered inputs in a batch
            
        # cut up into 2048* sample windows and apply Hann window
        # *wlen adjusted to fs to keep frequency resolution as similar to norm as possible
        xw_nw = torch.nn.functional.pad(x_cat, (0, int(np.ceil((x_cat.shape[1]-self.nfft)/self.step))*self.step+self.nfft-x_cat.shape[1])).unfold(dimension=1, size=self.nfft, step=self.step)
        xw_nw = xw_nw * torch.hann_window(self.nfft, periodic=False)

        # fft
        xw_nw = torch.fft.rfft(xw_nw, n=self.nfft, dim=2, norm='forward')
            # may be replaced with torch.stft()
            # ! ITU-R BS.1387-2 specifies normalization by 1/nfft
        #print(xw_nw.shape) # Batch x wnum x spectrum length

        # rectification
            # probably solved by the torch.abs() in weighting step?

        #scaling of the input signals
        #xw_nw = self.fac*xw_nw[:,:,:942]    # second half removal redundant when using rfft
        xw_nw=self.fac*xw_nw
        
        # outer and middle ear weighting function
        #f = np.linspace(0,self.fs//2,xw_nw.shape[-1],endpoint=False)
        f = torch.linspace(0,self.fs//2,xw_nw.shape[-1]+1)[:-1]
            # approximately f = k*23.4375
        #f = f[imin:imax]
        W = -0.6*3.64*torch.pow(f/1000,-0.8) + 6.5*torch.exp(-0.6*torch.square((f/1000)-3.3)) - (1e-3)*torch.pow(f/1000, 3.6)
        xw_nc = torch.abs(xw_nw)*torch.pow(10,W/20)
            # this should work fine

        # preparing error signal calculation
        noisep = torch.abs(xw_nw[self.bsize:,:,self.imin:self.imax])-torch.abs(xw_nw[:self.bsize,:,self.imin:self.imax])
        xw = torch.cat((xw_nc[:,:,self.imin:self.imax], noisep), dim=0)
        #F0 = 10*torch.log10(torch.abs(xw_nw[self.bsize:,:,imin:imax])/torch.abs(xw_nw[:self.bsize,:,imin:imax]))
        EHS_B = self.ehs(20*torch.log10(torch.abs(xw_nw[self.bsize:,:,:])/torch.abs(xw_nw[:self.bsize,:,:])))

        # critical band grouping
        xw = self.crit_group(torch.abs(torch.square(xw)))
        xw, noisep = torch.vsplit(xw, [2*self.bsize])
        
        # adding internal noise
        fc = wf.f_c()
        W = torch.pow(10, 0.4*0.364*(torch.pow(fc/1000, -0.8)))
        uep = xw + torch.matmul(torch.ones(xw.shape, dtype=torch.double), torch.diag(W))

        # frequency domain spreading
        uep = self.freq_smear(fc, uep)
            # changes dim to batch x frequency x time
        mp, mod_eline = self.calc_mod(uep)

        # time domain spreading
        tau = 0.008+(2.2/fc)
        a_vec = torch.exp(-4/(187.5*tau))
        af = torch.cat((torch.ones(a_vec.shape[0],1), -torch.unsqueeze(a_vec,-1)), -1)
        bf = torch.cat((1-torch.unsqueeze(a_vec,-1), torch.zeros(a_vec.shape[0],1)), -1)
        
        ep = torchaudio.functional.lfilter(torch.cat((torch.zeros(uep.shape[0],uep.shape[1],1), uep[:,:,1:]), dim=-1), af, bf, clamp=False)
        ep=torch.maximum(ep,uep)

        
        return ep, mp, mod_eline, xw_nw, noisep, EHS_B, W

    def crit_group(self, x):
        return torch.clamp(torch.matmul(x,self.barkmat), min=0.000000000001)
            # Batch x wnum x wlen*
            # *number of critical bands

    def freq_smear(self, fc, uep):
        L = 10*torch.log10(uep)
        Su_j0 = -24-(230/fc)
        Su = Su_j0+0.2*L
            # only works as long as number of bands is last dimension of L
        Sl = 27     # lower slope is a constant
        res = 0.25  # Bark scale resolution for basic version
        Z = 109     # maximum of j (number of frequency bands) 
        
        #mu = torch.arange(Z, dtype=torch.double)
        
        jmink = torch.empty(Z,Z, dtype=torch.double)
        jminkvals = []
        for i in range(Z):
            jminkvals.extend(list(range(Z-i)))
        jminkvals = torch.tensor(jminkvals, dtype=torch.double)
        i, j = torch.triu_indices(Z, Z)
        jmink[i,j] = -jminkvals
        jmink.T[i,j] = jminkvals
        Ecurl = torch.triu((torch.ones(Z,Z, dtype=torch.double)*Sl))+torch.tril(Su_j0.repeat(Z,1).transpose(0,1))
        Ecurl = torch.pow(10, torch.clamp((res*jmink*Ecurl)/10, max=10*np.log10(1.79769e308)))
        Ecurl = Ecurl/torch.sum(Ecurl, dim=1)

        # Su is shape [b, n, k] and must be transposed to be broadcastable
        Eline = torch.triu((torch.ones(Z,Z, dtype=torch.double)*Sl))+torch.tril(Su.repeat(Z,1,1,1).transpose(0,1).transpose(1,2).transpose(2,3))
        Eline = torch.pow(10, torch.clamp((res*jmink*Eline)/10, max=10*np.log10(1.79769e308)))*torch.pow(10, torch.clamp(L/10, max=10*np.log10(1.79769e308))).repeat(Z,1,1,1).transpose(0,1).transpose(1,2).transpose(2,3)
        Eline = Eline/torch.sum(Eline, dim=2).repeat(Z,1,1,1).transpose(0,1).transpose(1,2).transpose(2,3)
        
        NormSP_inv = 1/torch.pow(torch.clamp(torch.sum(torch.pow(Ecurl, 0.4), dim=1), max=(1.79769e308)**0.4), 1/0.4)
        
        E2 = torch.pow(torch.clamp(torch.sum(Eline**0.4, dim=2), max=(1.79769e308)**0.4), (1/0.4))*NormSP_inv

        return E2.transpose(1,2)

    def calc_mask(self, x):
        res = 0.25

        m = 3*torch.ones(x.shape, dtype=torch.double)
        m[:,int(np.ceil(12/res)):,:]=0.25*torch.arange(np.ceil(12/res),x.shape[1], dtype=torch.double).unsqueeze(-1)*res

        return x/torch.pow(10, torch.clamp(m/10, max=10*np.log10(1.79769e308)))

    def pat_adap_LP(self, x):
        fc = wf.f_c()
        tau = 0.008 + (4.2/fc)
        a = torch.exp(-self.step/(self.fs*tau))

        af = torch.cat((torch.ones(a.shape[0],1), -torch.unsqueeze(a,-1)), -1)
        bf = torch.cat((1-torch.unsqueeze(a,-1), torch.zeros(a.shape[0],1)), -1)

        return torchaudio.functional.lfilter(x, af, bf, clamp=False)
    
    def Lev_Corr(self, Pt, Pr):
        LC = torch.square(torch.sum(torch.sqrt(Pt*Pr), dim=1)/torch.sum(Pt, dim=1))
        #LC[LC>1] = 1/LC[LC>1]
        return LC.repeat(Pr.shape[1],1,1).transpose(1,0)
    
    def lev_adap(self, Px):
        Pt, Pr = torch.vsplit(Px, [self.bsize])
        LevCorr = self.Lev_Corr(Pt, Pr)
        LCor = torch.zeros(LevCorr.shape)
        LCor[LevCorr<1]=1
        TCor = (1-LCor)+LCor*LevCorr
        RCor = LCor+(1-LCor)/LevCorr

        return Pt*TCor, Pr*RCor
    
    def pat_adap(self, Et, Er):
        fc = wf.f_c()
        tau = 0.008 + (4.2/fc)
        a = torch.exp(-self.step/(self.fs*tau))

        af = torch.zeros(a.shape[0],Et.shape[-1], dtype=torch.double)
        af[:,0]=1
        bf = torch.pow(torch.clamp(a.repeat(Et.shape[-1],1).transpose(0,1), max=(1.79769e308)**(1/(Et.shape[-1]-1))), torch.arange(Et.shape[-1]))
        num = torchaudio.functional.lfilter(Et*Er, af, bf, clamp=False)
        den = torchaudio.functional.lfilter(Er*Er, af, bf, clamp=False)
        R = num/den
        
        Rcor = torch.zeros(R.shape)
        Rcor[R<1]=1
        Rt = Rcor+(1-Rcor)/R
        Rr = (1-Rcor)+Rcor*R
        #Rt = torch.ones(R.shape, dtype=torch.double)
        #Rr = torch.ones(R.shape, dtype=torch.double)
        #Rt[R>1] = 1/R[R>1]
        #Rr[R<1] = R[R<1]

        M = 8
        M1 = 3
        M2 = 4
        Rtl = []
        Rrl = []
        for k in range(M2):
            Rtl.append(torch.sum(Rt[:,Rr.shape[1]-M1-k-1:,:], dim=1)/(M1+k+1))
            Rrl.append(torch.sum(Rr[:,Rr.shape[1]-M1-k-1:,:], dim=1)/(M1+k+1))
        Rtf_end = torch.stack(Rtl[::-1], dim=1)
        Rrf_end = torch.stack(Rrl[::-1], dim=1)

        af = torch.zeros(M, dtype=torch.double)
        af[0]=1
        bf = torch.ones(M, dtype=torch.double)

        Rtf = torchaudio.functional.lfilter(Rt.transpose(1,2), af, bf, clamp=False)
        Rrf = torchaudio.functional.lfilter(Rr.transpose(1,2), af, bf, clamp=False)

        Mdiv = torch.clamp(torch.arange(Rtf.shape[-1]-M2)+M2+1, max=M)
        Rtf = (Rtf[:,:,M2:]/Mdiv).transpose(1,2)
        Rrf = (Rrf[:,:,M2:]/Mdiv).transpose(1,2)
        
        Rt = torch.cat((Rtf, Rtf_end), dim=1)
        Rr = torch.cat((Rrf, Rrf_end), dim=1)
        
        af = torch.cat((torch.ones(a.shape[0],1), -torch.unsqueeze(a,-1)), -1)
        bf = torch.cat((1-torch.unsqueeze(a,-1), torch.zeros(a.shape[0],1)), -1)
        PCt = torchaudio.functional.lfilter(Rt, af, bf, clamp=False)
        PCr = torchaudio.functional.lfilter(Rr, af, bf, clamp=False)
        
        return Et*PCt, Er*PCr
    
    def calc_mod(self, x):
        fc = wf.f_c()
        tau = 0.008 + (4.2/fc)
        a = torch.exp(-self.step/(self.fs*tau))

        af = torch.cat((torch.ones(a.shape[0],1), -torch.unsqueeze(a,-1)), -1)
        bf = torch.cat((1-torch.unsqueeze(a,-1), torch.zeros(a.shape[0],1)), -1)
        E2pow = torch.pow(x, 0.3)
        E2dif = (self.fs/self.step)*torch.abs(E2pow - torch.cat((torch.zeros(x.shape[0], x.shape[1], 1), E2pow[:,:,:-1]), 2))
        Eder = torchaudio.functional.lfilter(E2dif, af, bf, clamp=False)
        El = torchaudio.functional.lfilter(E2pow, af, bf, clamp=False)

        Mod = Eder/(1+(El/0.3))
        return Mod, El
    
    def calc_loud(self, E):
        f =  wf.f_c()
        Eth = torch.pow(10, torch.clamp(0.364*torch.pow(f/1000, -0.8), max=np.log10(1.79769e308)))
        s = torch.pow(10, (-2-2.05*torch.arctan(f/4000)-0.75*torch.arctan(torch.square(f/1600)))/10)
        N = 1.07664*torch.pow(Eth/(s*(1e4)), 0.23)*(torch.pow(1-s+(s*torch.transpose(E, 1,2))/Eth, 0.23)-1)
        return (25/N.shape[1])*torch.sum(torch.clamp(N, min=0), dim=2), Eth

    def calc_MOV(self, mp, r_modEl, Eth, t_Ep, r_Ep, sp_log, Pnoise, Mask, ep):
        # Need to calculate:
        #   WinModDiff1_B
        #   AvgModDiff1_B
        #   AvgModDiff2_B
        #   RmsNoiseLoud_B
        #   BandwidthRef_B
        #   BandwidthTest_B
        #   Total NMR_B
        #   RelDistFrames_B
        #   MFPD_B
        #   ADB_B
        #   EHS_B
        ret1 = 5*[0]
        ret2 = 5*[0]

        # mp = (t_mp, r_mp)
        MD1B = self.ModDiff(mp, 1, 1)
        TempWt = torch.sum(r_modEl.transpose(1,2)/(r_modEl.transpose(1,2)+100*torch.pow(Eth,0.3)), dim=2)
        ret1[3] = self.WinX(MD1B)
            # WinModDiff1_B
        ret2[0] = self.AvgX(MD1B, W=TempWt)
            # AvgModDiff1_B
        ret2[1] = self.AvgX(self.ModDiff(mp, 0.1, 0.01), W=TempWt)
            # AvgModDiff2_B
        st, sr = torch.vsplit(0.15*mp.transpose(1,2)+0.5, [self.bsize])
        #beta = torch.exp(-1.5*(t_Ep-r_Ep)/r_Ep)
        #NL = ((Eth/st)**0.23)*(((1+torch.clamp(st*t_Ep-sr*r_Ep,min=0)/(Eth+sr*r_Ep*beta))**0.23)-1)
        NL = torch.pow(Eth/st,0.23)*(torch.pow(1+torch.clamp(st*t_Ep.transpose(1,2)-sr*r_Ep.transpose(1,2),min=0)/(Eth+sr*r_Ep.transpose(1,2)*torch.exp(-1.5*(t_Ep-r_Ep)/r_Ep).transpose(1,2)),0.23)-1)
        ret2[2] = torch.mean(self.RmsX(NL), dim=-1)   # RmsNoiseLoud_B
        # not sure if temporal and spectral averaging is in correct order

        ret1[:2] = self.BWidth(sp_log)
            # BandWidthRef_B, BandWidthTest_B

        PoverM = Pnoise/Mask
        ret1[2] = 10*torch.log10(torch.sum(torch.sum(PoverM, dim=2)/Pnoise.shape[2], dim=1)/Pnoise.shape[1])
            # total NMR_B
        ret2[4] = self.RDF(10*torch.log10(PoverM))    # RelDistFrames_B

        ret2[3], ret1[4] = self.mfpd_adb(10*torch.log10(ep))
            # MFPD_B, ADB_B

        return ret1, ret2

    def ModDiff(self, xx, negWt, offset):
        xt, xr = torch.vsplit(xx, [self.bsize])
        if negWt != 1:
            w = torch.ones(xt.shape)
            w[xt<xr]=negWt
        else:
            w=1
        md = w*torch.abs(xt-xr)/(offset+xr)
        #return 100*torch.sum(md, dim=1)/xt.shape[1]
        retval = 100*torch.sum(md, dim=1)/xt.shape[1]
        return retval

    def BWidth(self, FL):
        FLTst, FLRef = torch.vsplit(FL, [self.bsize])
        ZeroThreshold = torch.amax(FLTst[:,:,921:], dim=2)
        ges = torch.ge(FLRef[:,:,:921], 10+ZeroThreshold.repeat(921,1,1).transpose(0,1).transpose(1,2)).float()
        BWRef=921-torch.argmax(torch.flip(ges, dims=[-1]), dim=2).float()
        
        gest = torch.ge(FLTst[:,:,:921], 5+ZeroThreshold.repeat(921,1,1).transpose(0,1).transpose(1,2)).float()
        tstmask = torch.arange(921) < BWRef.unsqueeze(-1)
        BWTst = 921-torch.argmax(torch.flip(gest*tstmask, dims=[-1]), dim=2).float()

        gts = torch.gt(BWRef, 346)
        BWR_B = torch.sum(BWRef*gts, dim=1)/torch.sum(gts, dim=-1)
        BWT_B = torch.sum(BWTst*gts, dim=1)/torch.sum(gts, dim=-1)
        return BWR_B, BWT_B
    
    def RDF(self, x):
        xm = torch.amax(x, dim=2)
        fms = torch.ge(xm, 1.5)
        return torch.sum(fms, dim=1)

    def mfpd_adb(self, E):
        tE, rE = torch.vsplit(E, [self.bsize])
        Lkn = 0.3*torch.maximum(rE,tE)+0.7*tE
        s = torch.le(Lkn,0)*1e30+torch.gt(Lkn, 0)*(5.95072*torch.pow(torch.clamp(6.39468/(Lkn+(Lkn<=0)), max=(1.79769e308)**(1/1.71332)), 1.71332)+(9.01033e-11)*torch.pow(torch.clamp(Lkn, max=(1.79769e308)**(1/4)), 4)+(5.05622e-6)*torch.pow(torch.clamp(Lkn, max=(1.79769e308)**1/3), 3)-0.00102438*torch.square(torch.clamp(Lkn, max=np.sqrt(1.79769e308)))+0.0550197*Lkn-0.198719)
        e = rE-tE
        b = (4*torch.gt(rE, tE)+6*torch.le(rE,tE)).to(dtype=torch.double)
        a = torch.pow(10, torch.clamp(-0.5213902276543247/b, max=np.log10(1.79769e308)))/s
        pc = 1-torch.pow(10, torch.clamp(-a*torch.pow(torch.clamp(e, max=torch.pow(1.79769e308,-b)), b), max=np.log10(1.79769e308)))
        qc = torch.abs(torch.trunc(e))/s
            # INT() is assumed to mean rounding towards zero
            # it has been noted in literature, that floor may be more appropriate
        Pc = 1-torch.prod(1-pc, dim=1)
        Qc = torch.sum(qc, dim=1)

        #c0 = 0.9**(941/1881)
        c0 = 0.9
            # should be 0.9**(step_size/(nfft/2))
            # since step_size is nfft/2, c0 is 0.9**1, so 0.9
        #c1 = 0.99**(941/1881)
            # c1 should be 0.99 for the ame reason c0 is 0.9
            # but page 63 specifies it to be 1
            # this leads to it not being needed at all

        af = torch.tensor([1, -c0], dtype=Pc.dtype)
        bf = torch.tensor([1-c0, 0], dtype=Pc.dtype)
        Pc_curl = torchaudio.functional.lfilter(Pc, af, bf, clamp=False)
        MFPD = torch.amax(torch.clamp(Pc_curl, min=0), dim=-1)
        
        Qsum = torch.sum(Qc, dim=1)
        n_dist = torch.sum(torch.gt(Pc, 0.5), dim=1)
        ADB = torch.gt(n_dist, 0)*(torch.gt(Qsum,0)*torch.log10((Qsum+torch.le(Qsum,0))/(n_dist+torch.le(n_dist, 0)))-0.5*torch.le(Qsum,0))
        
        return MFPD, ADB
            
    def ehs(self, x):
        maxlag = int(2**np.floor(np.log2((18000/self.fs)*(np.ceil(1024*(self.fs/48000))))))
            # should always be 256

        # apparently the lag is over bands and so is the spectrum calculation
        C_norm = torch.sum(x*x,dim=-1)
        
        C_list = []
        for lag in range(maxlag):
            C_list.append(torch.sum(x[:,:,:x.shape[2]-lag]*x[:,:,lag:], dim=-1) / C_norm)
        C = torch.stack(C_list, dim=-1)
        
        C = C*torch.hann_window(maxlag).repeat(x.shape[0],x.shape[1],1)
            # windowed by normalized Hann window
        C = C-torch.mean(C,dim=-1,keepdim=True)
            # removing DC component
        C_ft = 20*torch.log10(torch.abs(torch.fft.rfft(C, dim=-1, norm='forward')))
            # power spectrum using FFT
        
        # peak detection
        if C_ft.shape[-1]<4:                        # for extremely short spectra (should never happen)
            ehs=torch.max(C_ft,dim=-1)
        elif C_ft.shape[-1]<16:                     # for very short spectra (also should never happen)
            ehs=torch.max(C_ft[:,:,1:],dim=-1)
        else:                                       # otherwise
            if C_ft.shape[-1]>32:                   # should be always
                N=8
                if x.shape[1]>128:                  # also should be always
                    N=16
                C_ft_norm = torch.max(torch.abs(C_ft),dim=-1,keepdim=True).values
                xf = C_ft_norm*torchaudio.functional.filtfilt(C_ft/C_ft_norm,torch.cat((torch.ones(1), torch.zeros(N-1))),torch.ones(N)/N,clamp=False)
                sxd = torch.sign(torch.cat((torch.zeros(C_ft.shape[0], C_ft.shape[1], 1), xf[:,:,1:]-xf[:,:,:-1]),dim=-1))
            else:
                sxd  = torch.sign(torch.cat((torch.zeros(C_ft.shape[0], C_ft.shape[1], 1), C_ft[:,:,1:]-C_ft[:,:,:-1]),dim=-1))
            
            # diff of sign of diff
            sx2d = torch.cat((sxd[:,:,:-1]-sxd[:,:,1:], torch.zeros(C_ft.shape[0], C_ft.shape[1], 1)),dim=-1)

            # first valley index
            minidx = torch.argmax((sx2d==-2).to(dtype=torch.int),dim=-1)
            # first peak after minidx
            ehs = torch.empty(C_ft.shape[0],C_ft.shape[1])
            for b in range(C_ft.shape[0]):
                for t in range(C_ft.shape[1]):
                    pkidx = minidx[b,t]+torch.argmax((sx2d[b,t,minidx[b,t]:]==2).to(dtype=torch.int),dim=-1)
                    ehs[b,t] = x[b,t,pkidx]
        # as quoted from the norm:
        # "The average value of this maximum over frames multiplied by 1 000.0 is the error harmonic structure (EHS) variable."
        # maybe should be turned into torch.nanmean() later if energy thresholding is added
        return 1000*torch.mean(ehs, dim=-1)

    def AvgX(self, x, W=None):
        if W==None:
            N = x.shape[1]
        else:
            N = torch.sum(W)
            x = x*W
        return torch.sum(x, dim=1)/N
    
    def RmsX(self, x, W=None, Z=1):
        if W==None:
            N = x.shape[1]
        else:
            N = torch.sum(torch.square(W))
            x = x*W
        return np.sqrt(Z)*torch.sqrt(torch.clamp(torch.sum(torch.square(x), dim=1)/N, min=1e-30))

    def WinX(self, x):
        print('##################\nWinX only works for BxN shape tensors!\n##################')
        x_mean = torch.clamp(torch.mean(x.unfold(dimension=1, size=4, step=1), dim=-1), max=(1.79769e308)**0.25)
        return torch.sqrt(torch.clamp(torch.sum(torch.pow(x_mean, 4), dim=-1)/(x.shape[-1]-3), min=1e-12))


class PEMOQ(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        """
        Args:
            predictions (torch.Tensor): The model's predictions.
            targets (torch.Tensor): The ground truth labels or targets.

        Returns:
            torch.Tensor: The computed loss value.
        """

        # 0. delay compensation and level adjustment omitted
        # 1. auditory model
        predictions = self.auditory(predictions)
        targets = self.auditory(targets)

        # 2. assimilation

        # 3. cross correlation (PSM)

        PSM = self.xcorr(predictions, targets)

        # maybe 4. summing predictions, weighting, quantile

        return PSM
    
    def xcorr(self, x, y):
        """
        Calculate the cross-correlation between two tensors.

        Args:
            x (torch.Tensor): First input tensor.
            y (torch.Tensor): Second input tensor.

        Returns:
            torch.Tensor: Cross-correlation result.
        """
        # Ensure the tensors have the same shape
        if x.shape != y.shape:
            raise ValueError("Input tensors must have the same shape")

        # Normalize the tensors
        x = x - x.mean()
        y = y - y.mean()

        # Compute cross-correlation
        numerator = torch.sum(x * y)
        denominator = torch.sqrt(torch.sum(x ** 2) * torch.sum(y ** 2))
        return numerator / (denominator + 1e-8)  # Add epsilon to avoid division by zero
    
    def auditory(self, x):
        """
        Simulate the auditory model processing.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Processed tensor.
        """
        # basilar membrane filterbank
        x = torch.clamp(x, min=0) # halfwave rectification
        # low pass filter
        # absolute threshold
        # nonlinear adaptation
        # modulation filterbank
        return x
    

class ViSQOLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        """
        Args:
            predictions (torch.Tensor): The model's predictions.
            targets (torch.Tensor): The ground truth labels or targets.

        Returns:
            torch.Tensor: The computed loss value.
        """

        # 0. mid channel extraction zero padding removal, and power alignment omitted
        # 1. gammatone spectrogram
        predictions = self.gtone(predictions)
        targets = self.gtone(targets)
        # 2. patch creation
        
        # 3. patch and subpatch alignment (needs to be simplified, maybe even omitted)

        # 4. NSIM

        # 5. mean of mean per frequency?
        Q = NSIM.mean()

        return Q
    
    def gtone(self, x, num_channels=32, fs=44100, wlen=256, wstep=128):
        """
        Compute the gammatone spectrogram of the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_samples).
            num_channels (int): Number of gammatone filterbank channels. Default is 32.
            fs (int): Sampling frequency of the input signal in Hz. Default is 44100.
            wlen (int): Window length for spectrogram computation. Default is 256.
            wstep (int): Step size (hop length) for spectrogram computation. Default is 128.

        Returns:
            torch.Tensor: Gammatone spectrogram of shape (batch_size, num_channels, num_frames).
        """

        # Create gammatone filterbank (using a placeholder implementation)
        filters = torch.linspace(100, fs // 2, num_channels)  # Center frequencies

        # Apply gammatone filters
        spectrogram = []
        for center_freq in filters:
            # Simulate gammatone filtering (placeholder for actual filter implementation)
            filtered_signal = torch.abs(torch.fft.rfft(x * torch.cos(2 * torch.pi * center_freq * torch.arange(x.shape[1]) / fs)))
            spectrogram.append(filtered_signal)

        # Stack the spectrogram along the channel dimension
        spectrogram = torch.stack(spectrogram, dim=1)

        # Downsample to match window and hop size
        spectrogram = spectrogram.unfold(-1, wlen, wstep).mean(dim=-1)

        return spectrogram

if __name__=="__main__":
    torch.autograd.set_detect_anomaly(True)
    peaq = PEAQ()
    #sig = torch.randn((2,10000), dtype=torch.double)
    sig, fs = torchaudio.load('sample.wav')
    dsig = torch.sgn(sig)*torch.sqrt(torch.abs(sig))
    dsig = torch.cat((dsig,sig))
    dsig.requires_grad_(requires_grad=True)
    dsig.retain_grad()
    sig = sig.repeat(2,1)
    ls = peaq(sig, dsig)
    print(ls)
    bkw = ls.backward()
    print(bkw)
    print(dsig.grad)