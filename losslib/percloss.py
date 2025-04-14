import torch
import torchaudio

class myMSE(torch.nn.Module):
    def __init__(self, type=0):
        super(myMSE, self).__init__()
        self.type=type

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
        match self.type:
            case 0:
                # a simple preemphasis
                kernel = torch.tensor([0.25, 0.5, 0.25], device=x.device).view(1, 1, -1)
            case 1:
                # closer to A-weight
                kernel = torch.tensor([0.25, 0.5, 0.25], device=x.device).view(1, 1, -1)
            case 2:
                # outer and middle ear
                kernel = torch.tensor([0.25, 0.5, 0.25], device=x.device).view(1, 1, -1)
            case _:
                raise ValueError(f"Invalid loss type {self.type}.")
        x = torch.nn.functional.conv1d(x.unsqueeze(1), kernel, padding=1).squeeze(1)
        return x

class cepdist(torch.nn.Module):
    def __init__(self, type=0, p=2.0):
        super(myMSE, self).__init__()
        self.type=type
        self.p=p

    def forward(self, predictions, targets):
        predictions = self.cep(predictions)
        targets = self.cep(targets)

        return torch.cdist(predictions, targets, p=self.p) #may not be best, revisit when necessary
    
    def cep(self, x):
        spec_tf=torchaudio.transforms.Spectrogram(n_fft=128,
                                                  hop_length=64,
                                                  window_fn=torch.hann_window,
                                                  power=2,
                                                  normalized=False)
        X=spec_tf(x)
        match self.type:
            case 0:
                # normal power cepstrum
                Xl=torch.log(X)
                #ceps_tf=torchaudio.transforms.
                # will need to find a good way to do ifft on a spectrogram
            case 1:
                # MFCC, transform x into mel spectrum
                f_tf=torchaudio.transforms.MelScale(n_mels=80,
                                                    sample_rate=44100,
                                                    f_min=0.0,
                                                    f_max=None
                                                    n_stft=128, #same as nfft
                                                    norm=None,
                                                    scale='htk')
                X=f_tf(X)
                Xl=torch.log(X)
                #ceps_tf=torchaudio.transforms.
                # will need to test options for calculating dct
            #case 2:
                # PLPCC
            case _:
                raise ValueError(f"Invalid loss type {self.type}.")
        cx = ceps_tf(Xl)
        cx = torch.nn.functional.conv1d(x.unsqueeze(1), kernel, padding=1).squeeze(1)
        return x


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
