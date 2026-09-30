# SepRQ : Self-Supervised Speech Mixture Representation Learning Via Mask-Free Multi-Scale Source Separation

🗣️ SSL speech feature extractors for **SepRQ**, tailor-made for 🍸
cocktail-party 🍸 speech processing. Pulled from 🤗
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
