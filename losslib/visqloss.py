import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
import torchaudio
try:
    import losslib.wfilters as wf
except:
    import wfilters as wf

def erb(f_hz):
    # Glasberg & Moore (1990) equivalent rectangular bandwidth
    return 24.7 + f_hz / 9.265

def erb_space(f_min, f_max, n_filters):
    # center frequencies uniformly spaced on the ERB scale
    ear_q, min_bw = 9.26449, 24.7
    return -(ear_q * min_bw) + np.exp(
        np.arange(1, n_filters + 1) * (-np.log(f_max + ear_q * min_bw) +
                                        np.log(f_min + ear_q * min_bw)) / n_filters
    ) * (f_max + ear_q * min_bw)

def gammatone_filterbank_matrix(fs, nfft, n_chan, f_min=50.0, f_max=None, order=4):
    """Returns (n_filters, n_fft//2+1) numpy array of squared-magnitude gammatone
    responses evaluated at the STFT bin frequencies."""
    f_max = f_max or fs / 2
    cfs = erb_space(f_min, f_max, n_chan)            # center freqs, descending
    b = 1.019 * erb(cfs)                                # Slaney bandwidth per filter

    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)          # (n_fft//2+1,)
    # broadcast: (n_filters, 1) vs (1, n_bins)
    resp = (1.0 + ((freqs[None, :] - cfs[:, None]) / b[:, None]) ** 2) ** (-order)

    resp /= resp.max(axis=1, keepdims=True)             # normalize each filter to unit peak
    return resp.astype(np.float32)

class ViSQOLoss(torch.nn.Module):
    def __init__(self, fs):
        super().__init__()
        self.fs = fs
        self.nfft = int(round(0.080 * fs))
        self.nstep = int(round(0.020 * fs))
        self.gtonechan = 32

        gt_mat = gammatone_filterbank_matrix(self.fs, self.nfft, self.gtonechan, f_min=50)
        self.register_buffer("gt_mat", torch.from_numpy(gt_mat), persistent=False)
        self.register_buffer("window", torch.hann_window(self.nfft), persistent=False)


    def forward(self, predictions, targets):
        """
        Args:
            predictions (torch.Tensor): The model's predictions.
            targets (torch.Tensor): The ground truth labels or targets.

        Returns:
            torch.Tensor: The computed loss value.
        """
        #input check
        if predictions.shape != targets.shape:
            raise Exception("Error: Input dimensions do not match!")
        if predictions.ndim < 1 or predictions.ndim > 2:
            raise Exception("Error: Expected input of shape (,T) or (B,T)!")
        if predictions.ndim == 1:
            predictions = predictions.unsqueeze(0)
            targets = targets.unsqueeze(0)

        # 0.0. mid channel extraction and zero padding removal omitted
            # signal shape assumed to be Batch x Time
        # 1. power alignment (may ned to be solved somehow later)
        pred = self.pwr_align(predictions, targets)
        # 2. gammatone spectrogram
        Ggram_p, Ggram_t = self.gtone(pred, targets)
            # change to the following if low on memory during calculation
            # Ggram_p, Ggram_t = checkpoint(self.gtone, pred, targets, use_reentrant=False)
        # 3. patch creation
        
        # 4. patch and subpatch alignment (needs to be simplified, maybe even omitted)

        # 5. NSIM
        NSIM = self.calc_NSIM()
        # 6. mean of mean per frequency?
        Q = NSIM.mean()

        return Q

    def pwr_align(self, p, t):
        Pp = torch.mean(torch.square(torch.abs(p)), dim=-1, keepdim=True)
        Pt = torch.mean(torch.square(torch.abs(t)), dim=-1, keepdim=True)
        return torch.sqrt(torch.clamp(Pt, min=1e-12))*p/torch.sqrt(torch.clamp(Pp, min=1e-12))

    def gtone(self, p, t):
        p_spec = torch.stft(
            p, n_fft=self.nfft, hop_length=self.nstep, win_length=self.nfft,
            window=self.window.to(p.dtype), return_complex=True, center=True,
        )                                        # (..., nfft//2+1, frames)
        t_spec = torch.stft(
            t, n_fft=self.nfft, hop_length=self.nstep, win_length=self.nfft,
            window=self.window.to(t.dtype), return_complex=True, center=True,
        )
        P_pp = p_spec.real ** 2 + p_spec.imag ** 2
        P_tt = t_spec.real ** 2 + t_spec.imag ** 2

        Ggram_p = torch.einsum("cb,...bt->...ct", self.gt_mat.to(P_pp.dtype), P_pp)
        Ggram_t = torch.einsum("cb,...bt->...ct", self.gt_mat.to(P_tt.dtype), P_tt)

        Ggram_p_log = 10*torch.log10(torch.clamp(Ggram_p, min=1e-12))
        Ggram_t_log = 10*torch.log10(torch.clamp(Ggram_t, min=1e-12))

        floor = Ggram_t_log.amin(dim=(-2, -1), keepdim=True)
        Ggram_p_log = torch.clamp(Ggram_p_log, min=floor) - floor
        Ggram_t_log = Ggram_t_log - floor

        return Ggram_p_log, Ggram_t_log

    def calc_NSIM():
        pass
