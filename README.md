# SepRQ : Self-Supervised Speech Mixture Representation Learning Via Mask-Free Multi-Scale Source Separation

SSL speech feature extractors for **SepRQ** and **BEST-RQ (50 Hz)**. The
encoder (CNN frontend + 12-layer Conformer) and global norm stats are pulled
from the Hugging Face Hub ([SevKod/SepRQ](https://huggingface.co/SevKod/SepRQ))
on first use — only the requested model is downloaded. Upstream
`speechbrain>=1.0.3` only; runs on CPU or GPU.

## Install

```bash
pip install seprq
# or  from the git repo
pip install git+https://github.com/SevKod/SepRQ.git    # install straight from GitHub
```

## Use

```python
from seprq import SepRQEncoder

speech_encoder = SepRQEncoder("SepRQ", scale="reduced")   # model: "SepRQ" / "BestRQ_50Hz"
                                                          # scale: "reduced" (released) / "full" (TBA)
layers = speech_encoder("utterance.wav")       # list of 12 tensors, each [1, T, 576]
final = layers[-1]                             # last Conformer layer
```

Calling the encoder runs `forward` and returns the **12 Conformer layer
outputs** as a list, each `[1, T, 576]`. Only the **reduced**-scale checkpoints
are released for now; **full**-scale is TBA.

It accepts a **file path** (any format/sample rate — decoded, downmixed to mono
and resampled to 16 kHz for you) or a raw 1D 16 kHz waveform (numpy array / tensor).

The repo is private, so authenticate once: `huggingface-cli login` (or set `HF_TOKEN`).
