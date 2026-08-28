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

def erb_gammatone_coeffs(fs, n_chan, f_min=50.0, f_max=None):
    """Slaney ERB gammatone as a cascade of 4 biquads per channel.
    Returns b: (4, n_chan, 3), a: (n_chan, 3)."""
    f_max = f_max or fs / 2
    T = 1.0 / fs
    cf = erb_space(f_min, f_max, n_chan)
    B = 1.019 * 2 * np.pi * erb(cf)

    cos_, sin_, e = np.cos(2 * cf * np.pi * T), np.sin(2 * cf * np.pi * T), np.exp(B * T)
    sq_p, sq_m = np.sqrt(3 + 2 ** 1.5), np.sqrt(3 - 2 ** 1.5)

    A0 = T * np.ones_like(cf)
    A1 = np.stack([                                     # (4, n_chan)
        -(2 * T * cos_ / e + 2 * sq_p * T * sin_ / e) / 2,
        -(2 * T * cos_ / e - 2 * sq_p * T * sin_ / e) / 2,
        -(2 * T * cos_ / e + 2 * sq_m * T * sin_ / e) / 2,
        -(2 * T * cos_ / e - 2 * sq_m * T * sin_ / e) / 2,
    ])

    a = np.stack([np.ones_like(cf), -2 * cos_ / e, np.exp(-2 * B * T)], axis=-1)
    b = np.stack([np.broadcast_to(A0, A1.shape), A1, np.zeros_like(A1)], axis=-1)

    # unit gain at each center frequency, evaluated on the unit circle
    z = np.exp(2j * np.pi * cf / fs)
    zi = np.stack([np.ones_like(z), z ** -1, z ** -2]).T          # (n_chan, 3)
    H = np.prod((b * zi[None]).sum(-1), axis=0) / ((a * zi).sum(-1) ** 4)
    b = b / np.abs(H)[None, :, None] ** 0.25                       # spread across sections

    return b.astype(np.float32), a.astype(np.float32)

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
        Patch_p = self.patchify(Ggram_p)
        Patch_t = self.patchify(Ggram_t)
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
        Ggram_p = self._gtone(p)
        Ggram_t = self._gtone(t)

        floor = Ggram_t.amin(dim=(-2, -1), keepdim=True)
        Ggram_p_log = torch.clamp(Ggram_p, min=floor) - floor
        Ggram_t_log = Ggram_t - floor

        return Ggram_p_log, Ggram_t_log

    def _gtone(self, x):
        raise NotImplementedError

    def patchify(self, G, Plen=30, Phop=None):
        """(B, chan, T_in_frames) -> (B, T_in_patches, chan, Plen)"""
        Phop = Plen if Phop is None else Phop
        if G.shape[-1] < Plen:
            raise ValueError(f"need at least {Plen} frames, got {G.shape[-1]}")
        return G.unfold(-1, Plen, Phop).movedim(-2, -3) # movedim to make it (..., chan, Plen)

    def calc_NSIM(self):
        pass

class ViSQOLoss_f(ViSQOLoss):
    def __init__(self, fs):
        super().__init__(fs)
        gt_mat = gammatone_filterbank_matrix(self.fs, self.nfft, self.gtonechan, f_min=50)
        self.register_buffer("gt_mat", torch.from_numpy(gt_mat), persistent=False)
        self.register_buffer("window", torch.hann_window(self.nfft), persistent=False)

    def _gtone(self, x):
        X = torch.stft(
            x, n_fft=self.nfft, hop_length=self.nstep, win_length=self.nfft,
            window=self.window.to(x.dtype), return_complex=True, center=True,
        )
        P_xx = X.real ** 2 + X.imag ** 2
        Ggram = torch.einsum("cb,...bt->...ct", self.gt_mat.to(P_xx.dtype), P_xx)
        Ggram_log = 10*torch.log10(torch.clamp(Ggram, min=1e-12))
        return Ggram_log

class ViSQOLoss_t(ViSQOLoss):
    def __init__(self, fs):
        super().__init__(fs)
        b, a = erb_gammatone_coeffs(fs, self.gtonechan, f_min=50)
        self.register_buffer("b_coeffs", torch.from_numpy(b), persistent=False)
        self.register_buffer("a_coeffs", torch.from_numpy(a), persistent=False)

    def _gtone(self, x):
        y = x.unsqueeze(-2).expand(-1, self.gtonechan, -1)          # (B, chan, T)
        for k in range(4):
            y = torchaudio.functional.lfilter(
                y, self.a_coeffs.to(y.dtype), self.b_coeffs[k].to(y.dtype),
                clamp=False, batching=True,
            )
        pad = self.nfft//2
        y2 = torch.nn.functional.pad(y ** 2, (pad, pad), mode="reflect")    # padding only to match frequency domain shape
        Ggram = y2.unfold(-1, self.nfft, self.nstep).mean(dim=-1)   # (B, chan, T_in_frames)
        return 10*torch.log10(torch.clamp(Ggram, min=1e-12))