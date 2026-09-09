# seprq

SSL speech feature extractors for **SepRQ** and **BEST-RQ (50 Hz)**. The
encoder (CNN frontend + 12-layer Conformer) and global norm stats are pulled
from the Hugging Face Hub ([SevKod/SepRQ](https://huggingface.co/SevKod/SepRQ))
on first use — only the requested model is downloaded. Upstream
`speechbrain>=1.0.3` only; runs on CPU or GPU.

## Install

```bash
pip install git+https://github.com/SevKod/SepRQ.git    # install straight from GitHub
# or, from a local checkout:
pip install .
```

## Use

```python
from seprq import SepRQEncoder

speech_encoder = SepRQEncoder("SepRQ")         # or "BestRQ_50Hz"
feats = speech_encoder("utterance.wav")        # torch.Tensor [1, T, 576]
```

Calling the encoder runs `forward`. It accepts a **file path** (any
format/sample rate — decoded, downmixed to mono and resampled to 16 kHz for
you) or a raw 1D 16 kHz waveform (numpy array / tensor).

The repo is private, so authenticate once: `huggingface-cli login` (or set `HF_TOKEN`).
