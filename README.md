# SepRQ 🔀 👥 : Self-Supervised Speech Mixture Representation Learning Via Mask-Free Multi-Scale Source Separation

🗣️ SSL speech feature extractors for **SepRQ**, tailor-made for 🍸
cocktail-party 🍸 speech processing 🔀 👥. Pulled from 🤗
[SevKod/SepRQ](https://huggingface.co/SevKod/SepRQ) and ready to use; returns the
12 Conformer layer features. Requires **PyTorch** and upstream
`speechbrain>=1.0.3`; runs on CPU or GPU.

## Install

```bash
pip install seprq
```

## Use

```python
from seprq import SepRQEncoder

speech_encoder = SepRQEncoder("SepRQ")         # or "BestRQ_50Hz"
layers = speech_encoder("utterance.wav")       # list of 12 tensors, each [1, T, 576]
final = layers[-1]                             # last Conformer layer
```

Calling the encoder runs `forward` and returns the **12 Conformer layer
outputs** as a list, each `[1, T, 576]`.

It accepts a **file path** (any format/sample rate — decoded, downmixed to mono
and resampled to 16 kHz for you) or a raw 1D 16 kHz waveform (numpy array / tensor).

The repo is private, so authenticate once: `huggingface-cli login` (or set `HF_TOKEN`).

## 🔀 SepRQ — separation as the pretext task

SepRQ swaps masked prediction for pseudo source separation. It predicts one
stream of discrete units per speaker in the mixture, at several temporal
resolutions.

![SepRQ architecture](https://raw.githubusercontent.com/SevKod/SepRQ/main/assets/seprq_architecture.png)

*Fig. 1. SepRQ performs pseudo source separation in the discrete space. Random
vector quantizers (RVQs) give speaker-specific discrete labels at several
resolutions.*

- **Pseudo source separation** — each separation head predicts the frozen
  BEST-RQ codewords of one clean speaker from the mixture. Utterance-level PIT
  resolves speaker order, and no offline k-means is needed.
- **Mask-free** — masking can hide exactly the content needed to disentangle a
  mixture. SepRQ predicts units over the full utterance, which uses every frame
  for training.
- **Multi-resolution** — a separation objective follows every two Conformer
  layers, with frame folding going from 20 ms to 320 ms and one codebook per
  scale. Inference runs at 50 Hz with the encoder only.

## 👥 Results — state of the art on multi-speaker benchmarks

Frozen upstreams with SUPERB / TS-SUPERB downstream heads. SepRQ is pre-trained
on LibriSpeech 960 h mixtures only.

### Multi-speaker SUPERB

Speaker diarization (SD), speech separation (SS) and speech enhancement (SE).
LS = LibriSpeech, LL = Libri-Light, Mix = LL-60k + GigaSpeech-10k + VoxPopuli-24k.

| Model | Class | #Param (M) | Data (h) | DER ↓ | SI-SDRi ↑ | PESQ ↑ | STOI ↑ |
|-------|-------|-----------:|----------|------:|----------:|-------:|-------:|
| *Generic SSLs* | | | | | | | |
| HuBERT Base | Base | 94.68 | LS-960 | 5.88 | 9.36 | 2.58 | 93.9 |
| WavLM Base | Base | 94.70 | LS-960 | 4.55 | 10.37 | 2.58 | 94.0 |
| WavLM Base+ | Base | 94.70 | Mix-94k | 3.50 | 10.85 | 2.63 | 94.3 |
| HuBERT Large | Large | 316.62 | LL-60k | 5.75 | 10.45 | 2.64 | 94.2 |
| WavLM Large | Large | 316.62 | Mix-94k | 3.24 | 11.19 | 2.70 | 94.5 |
| *Cocktail-party SSLs* | | | | | | | |
| C-HuBERT Base | Base | 96.00 | LS-960 | 2.77 | 11.08 | 2.63 | 94.0 |
| C-HuBERT Large | Large | 318.00 | LL-60k | 2.65 | 11.24 | 2.65 | 94.3 |
| | | | | | | | |
| **⭐ SepRQ (ours)** | **Base** | **85.68** | **LS-960** | **2.08** | **12.10** | **2.67** | **94.4** |

Best values in bold.

### TS-SUPERB

Target speaker extraction (TSE), personalized speech extraction (PSE),
personalized VAD (PVAD), target-speaker ASR (TS-ASR).

| Upstream | #Param (M) | TSE<br>SI-SDRi ↑ | TSE<br>STOI ↑ | PSE<br>SI-SDRi ↑ | PSE<br>STOI ↑ | PVAD<br>mAP ↑ | TS-ASR<br>WER ↓ (w/o LM) | TS-ASR<br>WER ↓ (w/ LM) |
|----------|-----------:|----:|----:|----:|----:|----:|----:|----:|
| HuBERT Base | 94.68 | 9.64 | 87.30 | 8.61 | 79.92 | 94.60 | 36.86 | 30.52 |
| WavLM Base | 94.70 | 10.26 | 88.40 | 9.65 | 81.57 | 94.40 | 27.82 | 22.68 |
| WavLM Base+ | 94.70 | 10.69 | 89.00 | 10.01 | 82.67 | 95.00 | 24.75 | 20.06 |
| | | | | | | | | |
| **⭐ SepRQ (ours)** | **85.68** | **12.53** | **91.45** | **11.18** | **85.60** | **96.61** | **22.26** | **16.78** |

### Out-of-domain

EEND diarization on DIHARD 3 and ConvTasNet separation on WSJ0-2/3mix.

| Upstream | DIHARD 3 FA / MD / SC | DER ↓ | 2mix SDRi ↑ | 3mix SDRi ↑ |
|----------|-----------------------|------:|------------:|------------:|
| w/o SSL | 6.1 / 8.1 / 4.9 | 19.3 | 16.4 | 13.1 |
| HuBERT Base | 4.3 / 8.6 / 4.7 | 17.7 | 17.0 | 12.7 |
| WavLM Base | 4.4 / 8.3 / 4.5 | 17.3 | 17.5 | 13.2 |
| WavLM Base+ | 4.8 / 7.6 / 4.2 | 16.6 | 18.2 | 13.4 |
| **SepRQ (2 src)** | 4.9 / 7.4 / 3.8 | **16.1** | **19.8** | 14.5 |
| **SepRQ (3 src)** | 5.0 / 7.3 / 4.1 | 16.4 | 19.4 | **17.2** |
