import numpy as np
import torch
import torchaudio
try:
    import losslib.wfilters as wf
except:
    import wfilters as wf

class MSeE(torch.nn.Module):
    def __init__(self, mode=0, N=2047):
        super(MSeE, self).__init__()
        if mode in range(3):
            self.mode=mode
        else:
            raise ValueError(f"Invalid loss type {mode}.")
        match self.mode:
            case 0:
                # a simple first order pre-emphasis (simple high pass)
                self.a=torch.tensor([1, 0])
                self.b=torch.tensor([1, -0.85])
            case 1:
                # closer to A-weight
                # taken from a pdf online, but likely not very usable
                self.a=torch.tensor([1, -1.31861375911, 0.32059452332])
                self.b=torch.tensor([0.95616638497, -1.31960414122, 0.36343775625])
            case 2:
                # outer and middle ear
                # can be made by approximation of the weighting function in ITU_T BS.1387 pg. 35
                a=np.zeros(N)
                a[0]=1
                self.a=torch.from_numpy(a).type(torch.float)
                self.b=torch.from_numpy(wf.Acurve(N=N)).type(torch.float)
            case _:
                raise ValueError(f"Invalid loss type {self.mode}.")

    def forward(self, predictions, targets):
        """
        Args:
            predictions (torch.Tensor): The model's predictions.
            targets (torch.Tensor): The ground truth labels or targets.

        Returns:
            torch.Tensor: The computed MSeE loss value.
        """
        predictions = self.preem(predictions)
        targets = self.preem(targets)
        
        return torch.mean((predictions - targets)**2)
    
    def preem(self, x):
        #x = torch.nn.functional.conv1d(x.unsqueeze(1), kernel, padding=1).squeeze(1)
        x=torchaudio.functional.filtfilt(x,self.a,self.b,clamp=True) # time must be the last dim of x
        return x



class cepdist(torch.nn.Module):
    def __init__(self, mode='linear', wlen=128, wstep=64, fs=44100, p=2.0):
        super(cepdist, self).__init__()
        if mode in ['linear', 'mel', 'plp']:
            self.mode=['linear', 'mel', 'plp'].index(mode)
        else:
            raise ValueError(f"Invalid cepstrum type {mode}.")
        self.p=p
        self.wlen=wlen
        self.wstep=wstep
        self.fs=fs
        if self.mode:
            self.onesided=True
        else:
            self.onesided=False

    def forward(self, predictions, targets):
        predictions = self.cep(predictions)
        targets = self.cep(targets)

        distmat = torch.cdist(predictions, targets, p=self.p)

        return torch.mean(distmat, dim=(1,2))
    
    def cep(self, x):
        
        spec_tf=torchaudio.transforms.Spectrogram(n_fft=self.wlen,
                                                  hop_length=self.wstep,
                                                  window_fn=torch.hann_window,
                                                  power=2,
                                                  normalized=False,
                                                  onesided=self.onesided)
        X=spec_tf(x)
            # return a matrix with spectra in columns (wlen, wnum)
        match self.mode:
            case 0:
                # normal power cepstrum
                Xl=torch.log(X)
                if len(Xl.shape) == 2:
                    Xl = Xl.unsqueeze(0)
            case 1:
                mfcc = torchaudio.transforms.MFCC(sample_rate=self.fs,
                                                  n_mfcc=40,
                                                  dct_type=2,
                                                  norm='ortho',
                                                  log_mels=False,
                                                  melkwargs={'n_fft':self.wlen,
                                                             'hop_length':self.wstep,
                                                             'n_mels':40,
                                                             'center':False})
            #case 2:
                # PLPCC
            case _:
                raise ValueError(f"Invalid loss type {self.mode}.")
        if self.mode==0:
            cx = torch.real(torch.fft.ifft(Xl, n=None, dim=1, norm="backward"))
            cx = cx[:,0:(self.wlen//2)+1,:]
        elif self.mode==1:
            cx=mfcc(x)
        else:
            cx = torch.transpose(torch.matmul(torch.transpose(torch.squeeze(Xl), 0, 1), ceps_tf), 0, 1)
        return cx


class PEMOQ(torch.nn.Module):
    def __init__(self):
        super(PEMOQ, self).__init__()

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
        super(ViSQOLoss, self).__init__()

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
    cd = cepdist(mode='linear')
    print(cd(torch.randn(256), torch.randn(256)))
    print(cd)