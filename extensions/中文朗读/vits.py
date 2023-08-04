import time
import os
from scipy.io import wavfile
import asyncio
import subprocess
import requests
import json
import argparse
import torch
from torch import no_grad, LongTensor
import extensions.中文朗读.utils as utils
from extensions.中文朗读.models import SynthesizerTrn
from extensions.中文朗读.text import text_to_sequence
import extensions.中文朗读.commons as commons


vitsNoiseScale = 0.6
vitsNoiseScaleW = 0.668
vitsLengthScale = 1.2

_init_vits_model = False

hps_ms = None
device = None
net_g_ms = None
speakers = None

PATH = os.path.dirname(os.path.abspath(__file__))


def init_vits_model():
    global hps_ms, device, net_g_ms, speakers

    device = torch.device("cpu")

    hps_ms = utils.get_hparams_from_file(os.path.join(PATH, "./model/config.json"))
    net_g_ms = SynthesizerTrn(
        len(hps_ms.symbols),
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=hps_ms.data.n_speakers,
        **hps_ms.model,
    )
    _ = net_g_ms.eval().to(device)
    speakers = hps_ms.speakers
    for filename in os.listdir(os.path.join(PATH, "./model")):
        if filename.endswith(".pth"):
            model_path = os.path.join(PATH, "./model", filename)
            break
    model, optimizer, learning_rate, epochs = utils.load_checkpoint(
        model_path, net_g_ms, None
    )
    _init_vits_model = True


def get_text(text, hps):
    text_norm, clean_text = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm, clean_text


def vits(text, language, speaker_id, noise_scale, noise_scale_w, length_scale):
    start = time.perf_counter()
    if not len(text):
        return "输入文本不能为空！", None, None
    text = text.replace("\n", " ").replace("\r", "").replace(" ", "")
    if len(text) > 200:
        text = text[:200]
    if language == 0:
        text = f"[ZH]{text}[ZH]"
    elif language == 1:
        text = f"[JA]{text}[JA]"
    else:
        text = f"{text}"
    stn_tst, clean_text = get_text(text, hps_ms)
    with no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
        speaker_id = LongTensor([speaker_id]).to(device)
        audio = (
            net_g_ms.infer(
                x_tst,
                x_tst_lengths,
                sid=speaker_id,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )

    return "生成成功!", (22050, audio), f"生成耗时 {round(time.perf_counter()-start, 2)} s"


if not _init_vits_model:
    init_vits_model()
