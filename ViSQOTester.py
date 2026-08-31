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

    print(msg, end='')
    sd.play(dist.numpy().transpose(), fs)
    sd.wait()

    vlf_vals = vlossf(dist, raw)
    vlt_vals = vlosst(dist, raw)

    api = visqol_lib_py.VisqolApi()
    api.Create(config)

    similarity_result = api.Measure(raw.double().numpy().squeeze(), dist.double().numpy().squeeze())

    print(similarity_result.moslqo)

