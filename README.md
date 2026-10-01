# SepRQ 🔀 👥 : Self-Supervised Speech Mixture Representation Learning Via Mask-Free Multi-Scale Source Separation

🗣️ This framework is intended to provide an easy, ready-to-use and strong
**SepRQ** feature extractor for 🍸 cocktail-party 🍸 and multi-talker
conversational scenarios 🔀 👥, while also facilitating the use of other feature
extractors from the literature. We also release a **BEST-RQ** model trained at
50 Hz as a valuable contribution to the community.

SepRQ features are pulled from 🤗
[SevKod/SepRQ](https://huggingface.co/SevKod/SepRQ); a call returns the 12
Conformer layer features. Requires **PyTorch** and upstream `speechbrain>=1.0.3`;
runs on CPU or GPU.

---

👉 Please visit **[https://sevkod.github.io/SepRQ/](https://sevkod.github.io/SepRQ/)** for the full usage and demo.

---

## Install

```bash
pip install seprq
```

## Usage (frozen ❄️ or fine-tuned 🔥)

```python
from seprq import SepRQEncoder

encoder = SepRQEncoder("SepRQ", streams=2)   # streams=2|3 (SepRQ); or "BestRQ_50Hz", "WavLM_BASE", ... (see below)

# ❄️ frozen feature extraction
encoder.requires_grad_(False).eval()
layers = encoder(wavs)                      # wavs: [B, num_samples] or [B, channel, num_samples]

# 🔥 fine-tuning (plug into your model)
encoder.requires_grad_(True).train()
layers = encoder(wavs)
```

It accepts a **file path** (any format/rate — decoded, mono, resampled to 16 kHz)
or a **waveform tensor** `[num_samples]`, `[batch, num_samples]` or
`[batch, channel, num_samples]`; pass `wav_lens` (`[batch]`) for padded batches.

**Available models**

| name | streams | layers | dim |
|------|---------|-------:|----:|
| ⭐ `SepRQ` ⭐ | 2 or 3 | 12 | 576 |
| `BestRQ_50Hz` | — | 12 | 576 |
| `HuBERT_BASE` | — | 12 | 768 |
| `WavLM_BASE` | — | 12 | 768 |
| `WavLM_BASE_PLUS` | — | 12 | 768 |
| `WavLM_LARGE` | — | 24 | 1024 |

`streams` only applies to `SepRQ` (pick the 2- or 3-speaker variant); it is
ignored by the other models.

The SepRQ / BEST-RQ weights come from the Hub (private — authenticate once with
`huggingface-cli login` or `HF_TOKEN`); the HuBERT / WavLM baselines are fetched
from the torchaudio pipelines, nothing extra to host.

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

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2">Class</th>
      <th rowspan="2">#Param (M)</th>
      <th rowspan="2">Data (h)</th>
      <th>SD</th>
      <th>SS</th>
      <th colspan="2">SE</th>
    </tr>
    <tr>
      <th>DER ↓</th>
      <th>SI-SDRi ↑</th>
      <th>PESQ ↑</th>
      <th>STOI ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr><td colspan="8"><i>Generic SSLs</i></td></tr>
    <tr><td>HuBERT Base</td><td>Base</td><td>94.68</td><td>LS-960</td><td>5.88</td><td>9.36</td><td>2.58</td><td>93.9</td></tr>
    <tr><td>WavLM Base</td><td>Base</td><td>94.70</td><td>LS-960</td><td>4.55</td><td>10.37</td><td>2.58</td><td>94.0</td></tr>
    <tr><td>WavLM Base+</td><td>Base</td><td>94.70</td><td>Mix-94k</td><td>3.50</td><td>10.85</td><td>2.63</td><td>94.3</td></tr>
    <tr><td>HuBERT Large</td><td>Large</td><td>316.62</td><td>LL-60k</td><td>5.75</td><td>10.45</td><td>2.64</td><td>94.2</td></tr>
    <tr><td>WavLM Large</td><td>Large</td><td>316.62</td><td>Mix-94k</td><td>3.24</td><td>11.19</td><td>2.70</td><td>94.5</td></tr>
    <tr><td colspan="8"><i>Cocktail-party SSLs</i></td></tr>
    <tr><td>C-HuBERT Base</td><td>Base</td><td>96.00</td><td>LS-960</td><td>2.77</td><td>11.08</td><td>2.63</td><td>94.0</td></tr>
    <tr><td>C-HuBERT Large</td><td>Large</td><td>318.00</td><td>LL-60k</td><td>2.65</td><td>11.24</td><td>2.65</td><td>94.3</td></tr>
    <tr><td colspan="8"></td></tr>
    <tr><td><b>⭐ SepRQ ⭐</b></td><td><b>Base</b></td><td><b>85.68</b></td><td><b>LS-960</b></td><td><b>2.08</b></td><td><b>12.10</b></td><td><b>2.67</b></td><td><b>94.4</b></td></tr>
  </tbody>
</table>

### TS-SUPERB

Target speaker extraction (TSE), personalized speech extraction (PSE),
personalized VAD (PVAD), target-speaker ASR (TS-ASR).

<table>
  <thead>
    <tr>
      <th rowspan="3">Upstream</th>
      <th rowspan="3">#Param (M)</th>
      <th colspan="2">TSE</th>
      <th colspan="2">PSE</th>
      <th rowspan="3">PVAD<br>mAP ↑</th>
      <th colspan="2">TS-ASR</th>
    </tr>
    <tr>
      <th rowspan="2">SI-SDRi ↑</th>
      <th rowspan="2">STOI ↑</th>
      <th rowspan="2">SI-SDRi ↑</th>
      <th rowspan="2">STOI ↑</th>
      <th colspan="2">WER ↓</th>
    </tr>
    <tr>
      <th style="white-space: nowrap;">(w/o LM)</th>
      <th style="white-space: nowrap;">(w/ LM)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>HuBERT Base</td><td>94.68</td><td>9.64</td><td>87.30</td><td>8.61</td><td>79.92</td><td>94.60</td><td>36.86</td><td>30.52</td></tr>
    <tr><td>WavLM Base</td><td>94.70</td><td>10.26</td><td>88.40</td><td>9.65</td><td>81.57</td><td>94.40</td><td>27.82</td><td>22.68</td></tr>
    <tr><td>WavLM Base+</td><td>94.70</td><td>10.69</td><td>89.00</td><td>10.01</td><td>82.67</td><td>95.00</td><td>24.75</td><td>20.06</td></tr>
    <tr><td><b>⭐ SepRQ ⭐</b></td><td><b>85.68</b></td><td><b>12.53</b></td><td><b>91.45</b></td><td><b>11.18</b></td><td><b>85.60</b></td><td><b>96.61</b></td><td><b>22.26</b></td><td><b>16.78</b></td></tr>
  </tbody>
</table>

### Out-of-domain

EEND diarization on DIHARD 3 and ConvTasNet separation on WSJ0-2/3mix.

<table>
  <thead>
    <tr>
      <th rowspan="3">Upstream</th>
      <th colspan="2">Diarization</th>
      <th colspan="2">Separation</th>
    </tr>
    <tr>
      <th colspan="2">DIHARD 3</th>
      <th>WSJ0-2mix</th>
      <th>WSJ0-3mix</th>
    </tr>
    <tr>
      <th>FA / MD / SC ↓</th>
      <th>DER ↓</th>
      <th>SDRi ↑</th>
      <th>SDRi ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>w/o SSL features</td><td>6.1 / 8.1 / 4.9</td><td>19.3</td><td>16.4</td><td>13.1</td></tr>
    <tr><td>HuBERT Base</td><td>4.3 / 8.6 / 4.7</td><td>17.7</td><td>17.0</td><td>12.7</td></tr>
    <tr><td>WavLM Base</td><td>4.4 / 8.3 / 4.5</td><td>17.3</td><td>17.5</td><td>13.2</td></tr>
    <tr><td>WavLM Base+</td><td>4.8 / 7.6 / 4.2</td><td>16.6</td><td>18.2</td><td>13.4</td></tr>
    <tr><td><b>⭐ SepRQ (2 src) ⭐</b></td><td><b>4.9 / 7.4 / 3.8</b></td><td><b>16.1</b></td><td><b>19.8</b></td><td><b>14.5</b></td></tr>
    <tr><td><b>⭐ SepRQ (3 src) ⭐</b></td><td><b>5.0 / 7.3 / 4.1</b></td><td><b>16.4</b></td><td><b>19.4</b></td><td><b>17.2</b></td></tr>
  </tbody>
</table>

## Citation

> 🚧 **TODO** — BibTeX coming soon.

```bibtex
% TODO: citation to be added
```
