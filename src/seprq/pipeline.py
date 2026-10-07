"""SUPERB downstream heads on top of a frozen SepRQEncoder.

Each head is the s3prl checkpoint trained for one (task, upstream) pair.
The Hub file stores the learned 13-way layer weights and the downstream
module, without the optimizer state.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

SUPERB_TASKS = ("separation-2spk", "diarization", "enhancement")
SUPERB_UPSTREAMS = ("SepRQ", "HuBERT_BASE", "WavLM_BASE", "WavLM_BASE_PLUS")


class _SepRNN(nn.Module):
    """Mask estimator used by the SUPERB separation and enhancement heads."""

    def __init__(self, input_dim, num_bins, num_spks, num_layers, hidden_size, dropout, non_linear, bidirectional):
        super().__init__()
        self.rnn = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        out = hidden_size * 2 if bidirectional else hidden_size
        self.drops = nn.Dropout(dropout)
        self.linear = nn.ModuleList([nn.Linear(out, num_bins) for _ in range(num_spks)])
        self.non_linear = {
            "relu": torch.nn.functional.relu,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
        }[non_linear]
        self.num_bins = num_bins

    def forward(self, packed):
        x, _ = self.rnn(packed)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.drops(x)
        return [self.non_linear(linear(x)) for linear in self.linear]


class _DiarizationHead(nn.Module):
    def __init__(self, input_dim, hidden_size, num_speakers, rnn_layers):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_size, num_layers=rnn_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_speakers)

    def forward(self, features):
        hidden, _ = self.rnn(features.float())
        return self.linear(hidden)


def _match_length(feat, n_frames):
    """Trim or pad a [T, D] feature so it lines up with an STFT of n_frames."""
    if abs(feat.size(0) - n_frames) >= 5:
        raise RuntimeError(
            f"SSL frames ({feat.size(0)}) and STFT frames ({n_frames}) differ by 5 or more"
        )
    if feat.size(0) == n_frames:
        return feat
    if feat.size(0) > n_frames:
        return feat[:n_frames]
    out = feat.new_zeros(n_frames, feat.size(1))
    out[: feat.size(0)] = feat
    return out


def _suppress_impulse(x):
    """Zero the tail impulse the STFT mask sometimes leaves at the end of a wav."""
    y = np.copy(x)
    nonzero = np.nonzero(y)[0]
    if len(nonzero) == 0:
        return x
    p = int(np.max(nonzero)) + 1
    if p < x.shape[0] - 2048:
        return x
    window_size = 512
    start = p - window_size
    if start <= 0:
        return x
    max_value = np.max(np.abs(y[:start]))
    invalid = np.nonzero(np.abs(y[start:p]) > max_value)[0]
    if len(invalid) == 0:
        return x
    z = np.copy(x)
    z[int(np.min(invalid)) + start :] = 0
    return z


class SepRQPipeline(nn.Module):
    """Run a SUPERB downstream head on top of one upstream.

    .. code-block:: python

        pipe = SepRQPipeline("separation-2spk", upstream="SepRQ", streams=2)
        sources = pipe("mixture.wav")          # one waveform per speaker

        pipe = SepRQPipeline("diarization", upstream="HuBERT_BASE")
        activations = pipe("mixture.wav")      # [T, num_speakers]

        pipe = SepRQPipeline("enhancement", upstream="WavLM_BASE")
        clean = pipe("noisy.wav")              # enhanced waveform
    """

    def __init__(self, task, upstream="SepRQ", streams=2, repo_id=None, device=None):
        super().__init__()
        from seprq import REPO_ID, SepRQEncoder

        if task not in SUPERB_TASKS:
            raise ValueError(
                f"task must be one of {list(SUPERB_TASKS)}, got {task!r}"
            )
        if upstream == "SepRQ" and streams != 2:
            raise ValueError(
                "SUPERB heads are published for SepRQ with streams=2. "
                f"Got streams={streams!r}."
            )
        if upstream not in SUPERB_UPSTREAMS:
            raise ValueError(
                f"No SUPERB head is published for {upstream!r} on {task!r}. "
                f"Available upstreams: {list(SUPERB_UPSTREAMS)}."
            )

        self.task = task
        self.upstream_name = upstream
        self.repo_id = repo_id or REPO_ID
        self.encoder = SepRQEncoder(upstream, streams=streams, repo_id=self.repo_id, device=device)
        self.encoder.requires_grad_(False).eval()

        from huggingface_hub import hf_hub_download

        path = hf_hub_download(self.repo_id, f"superb/{task}/{upstream}/head.ckpt")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.n_states = int(blob["featurizer"].numel())
        self.register_buffer("layer_weights", blob["featurizer"].float())
        self._build_head(blob)
        self.to(self.encoder._device())
        self.eval()

    def _build_head(self, blob):
        weights = blob["model"]
        if self.task == "diarization":
            self.head = _DiarizationHead(
                blob["input_dim"], blob["hidden_size"], blob["num_speakers"], blob["rnn_layers"]
            )
        else:
            self.hop_length = int(blob["hop_length"])
            self.n_fft = int(blob["n_fft"])
            self.win_length = int(blob["win_length"])
            self.window = blob["window"]
            self.center = bool(blob["center"])
            self.num_speakers = int(blob["num_speakers"])
            self.head = _SepRNN(
                blob["input_dim"],
                self.n_fft // 2 + 1,
                self.num_speakers,
                blob["rnn_layers"],
                blob["hidden_size"],
                blob["dropout"],
                blob["non_linear"],
                blob["bidirectional"],
            )
        self.head.load_state_dict(weights)

    def _weighted(self, states):
        if len(states) != self.n_states:
            raise RuntimeError(
                f"The {self.upstream_name} head expects {self.n_states} hidden states, "
                f"got {len(states)}."
            )
        stacked = torch.stack([s.float() for s in states], dim=0)
        weights = torch.softmax(self.layer_weights, dim=-1)
        return (weights.view(-1, 1, 1, 1) * stacked).sum(dim=0)

    def _separate(self, wav):
        """wav: [N] on device. Returns a list of [N] waveforms."""
        import librosa

        audio = wav.detach().cpu().numpy().astype(np.float32)
        spec = np.transpose(
            librosa.stft(
                audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=self.center,
            )
        )
        feat = self._weighted(self.encoder.hidden_states(wav))[0]
        feat = _match_length(feat, spec.shape[0])
        masks = self.head(pack_sequence([feat]))
        waves = []
        mix = torch.from_numpy(spec)
        for mask in masks:
            masked = (mask[0].detach().cpu() * mix).numpy()
            wave = librosa.istft(
                np.transpose(masked),
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=self.center,
                length=audio.shape[0],
            )
            waves.append(
                torch.from_numpy(np.ascontiguousarray(_suppress_impulse(wave))).float().to(wav.device)
            )
        return waves

    def forward(self, audio, wav_lens=None):
        """Run the selected SUPERB task.

        audio is a path (local or ``hf.co/...``) or a 16 kHz waveform, same as
        ``SepRQEncoder``. One utterance returns the shapes used on the website:
        a list of waveforms, one enhanced waveform, or ``[T, num_speakers]``
        probabilities. A batch keeps a leading batch dimension.
        """
        batch = self.encoder._as_batch(audio)
        if self.task == "diarization":
            states = self.encoder.hidden_states(batch, wav_lens)
            logits = self.head(self._weighted(states))
            prob = torch.sigmoid(logits)
            return prob[0] if prob.shape[0] == 1 else prob

        outs = [self._separate(batch[i]) for i in range(batch.shape[0])]
        if self.task == "enhancement":
            waves = [item[0] for item in outs]
            return waves[0] if len(waves) == 1 else torch.stack(waves)
        if len(outs) == 1:
            return outs[0]
        return [torch.stack([item[s] for item in outs]) for s in range(self.num_speakers)]
