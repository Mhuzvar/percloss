import losslib.visqloss as vl
import torch
import torchaudio
import sounddevice as sd
import os
from visqol import visqol_lib_py
from visqol.pb2 import visqol_config_pb2
from visqol.pb2 import similarity_result_pb2
import numpy as np

PLAY_REF = False
PLAY_DIST = False
BITC = True

config = visqol_config_pb2.VisqolConfig()
config.audio.sample_rate = 48000
config.options.use_speech_scoring = False
svr_model_path = "libsvm_nu_svr_model.txt"
config.options.svr_model_path = os.path.join(
    os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)

def bit_crush(dry, lvls=None, *, bit_depth=8):
    if not lvls:
        lvls = 2**bit_depth
    wet = torch.round(dry*lvls)/lvls
    return wet



if __name__=="__main__":
    msg = 'Now playing distorted version with the following effects:\n'
    bitd = 4

    vlosst = vl.ViSQOLoss_t()
    vlossf = vl.ViSQOLoss_f()

    raw, fs = torchaudio.load('sample_long.wav')
    dist = torch.clone(raw)
    
    if BITC:
        dist = bit_crush(dist, bit_depth=bitd)
        msg+=f'quantization to {bitd} bits\n'

    if PLAY_REF:
        print("Now playing the original")
        sd.play(raw.numpy().transpose(), fs)
        sd.wait()

    if PLAY_DIST:
        print(msg, end='')
        sd.play(dist.numpy().transpose(), fs)
        sd.wait()

    vlf_vals = vlossf(dist, raw)
    vlt_vals = vlosst(dist, raw)

    api = visqol_lib_py.VisqolApi()
    api.Create(config)

    similarity_result = api.Measure(raw.double().numpy().squeeze(), dist.double().numpy().squeeze())

    Ggram_p, Ggram_t = vlosst.gtone(vlosst.pwr_align(dist, raw), raw)
    vl_NSIM = vlosst.calc_NSIM(vlosst.patchify(Ggram_p), vlosst.patchify(Ggram_t))
    vl_meanperf = torch.mean(vl_NSIM, dim=(-1,-3))

    #print(Ggram_t.amax(dim=(-2, -1)))

    #print(f"reference f vector: {similarity_result.fvnsim}\n\nestimated: {vl_meanperf}")

    '''    
    print(similarity_result)

    print(similarity_result.DESCRIPTOR.fields_by_name.keys())

    print([n for n in dir(visqol_lib_py) if not n.startswith('_')])

    print((Ggram_p - Ggram_t).abs().mean(dim=-1))   # per-band mean |Δ| in dB
    print(Ggram_t.mean(dim=-1))                      # reference level per band
    print(Ggram_p.mean(dim=-1))                      # degraded level per band

    imp = torch.zeros(1, 48000); imp[0, 0] = 1.0
    h = imp.unsqueeze(-2).expand(-1, 32, -1)
    for k in range(4):
        h = torchaudio.functional.lfilter(h, vlosst.a_coeffs, vlosst.b_coeffs[k], clamp=False)
    H = torch.fft.rfft(h).abs()
    print(H.argmax(dim=-1))     # peak bin per channel -> should track center_freq_bands
    print(H.amax(dim=-1))       # should be ~1.0 for every channel
    

    P_p = Ggram_p[..., 16:46].unsqueeze(1)      # (B, 1, chan, 30)
    P_t = Ggram_t[..., 16:46].unsqueeze(1)
    nsim = vlosst.calc_NSIM(P_p, P_t)
    print(nsim.mean(dim=(-1, -3)))
    print(list(similarity_result.patch_sims[0].freq_band_means))
    '''

    FRAME_S = vlosst.nstep / vlosst.fs          # 0.02 s per frame
    PLEN = 30

    ref = torch.tensor([list(ps.freq_band_means) for ps in similarity_result.patch_sims])
    starts = [round(ps.ref_patch_start_time / FRAME_S) for ps in similarity_result.patch_sims]

    def band_vectors(offset):
        """Per-patch band means with every ViSQOL patch shifted by `offset` frames."""
        idx = [s + offset for s in starts]
        if min(idx) < 0 or max(idx) + PLEN > Ggram_t.shape[-1]:
            return None
        P_p = torch.stack([Ggram_p[..., i:i+PLEN] for i in idx], dim=1)   # (B, P, chan, PLEN)
        P_t = torch.stack([Ggram_t[..., i:i+PLEN] for i in idx], dim=1)
        return vlosst.calc_NSIM(P_p, P_t).mean(dim=-1)[0]                 # (P, chan')

    results = []
    for off in range(-16, 17):
        est = band_vectors(off)
        if est is None:
            continue
        r = ref if est.shape[-1] == ref.shape[-1] else ref[:, 1:-1]
        d = est - r
        results.append((off, d.abs().mean().item(), d.mean().item(), d.std().item()))
        print(f"offset {off:+d}:  MAE {results[-1][1]:.4f}   bias {results[-1][2]:+.4f}   spread {results[-1][3]:.4f}")

    best = min(results, key=lambda t: t[1])
    print(f"\nbest offset {best[0]:+d}  (MAE {best[1]:.4f})")

    est = band_vectors(best[0])
    r = ref if est.shape[-1] == ref.shape[-1] else ref[:, 1:-1]
    print("per-band mean error:", (est - r).mean(dim=0))
