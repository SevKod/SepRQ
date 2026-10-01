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

![Multi-speaker SUPERB results](https://raw.githubusercontent.com/SevKod/SepRQ/main/assets/tab2_superb.png)

### TS-SUPERB

Target speaker extraction (TSE), personalized speech extraction (PSE),
personalized VAD (PVAD), target-speaker ASR (TS-ASR).

![TS-SUPERB results](https://raw.githubusercontent.com/SevKod/SepRQ/main/assets/tab3_ts_superb.png)

### Out-of-domain

EEND diarization on DIHARD 3 and ConvTasNet separation on WSJ0-2/3mix.

![Out-of-domain results](https://raw.githubusercontent.com/SevKod/SepRQ/main/assets/tab4_ood.png)
