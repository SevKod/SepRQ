"""SUPERB downstream heads on top of a frozen SepRQEncoder.

Each head is the s3prl checkpoint trained for one (task, upstream) pair.
The Hub file stores the learned 13-way layer weights and the downstream
module, without the optimizer state.
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

from .targets import TARGET_TASKS, build_target_head

SUPERB_TASKS = ("separation", "diarization", "enhancement")
TASKS = SUPERB_TASKS + TARGET_TASKS
# The SUPERB separation head is the 2-speaker setup. Checkpoints stay in the
# folder they were uploaded under.
_HUB_TASK = {"separation": "separation-2spk"}
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


def _is_path(value):
    return isinstance(value, (str, bytes)) or hasattr(value, "__fspath__")


def _item_dir(root, index, count):
    folder = root if count == 1 else root / str(index)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _write_wav(path, wave):
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wave.detach().cpu().numpy(), 16000, subtype="PCM_16")


def _rttm(prob, frame_shift, uri):
    """Collapse frame posteriors into an RTTM. ``prob`` is ``[time, speakers]``."""
    hop = frame_shift / 16000
    active = prob.detach().cpu() >= 0.5
    lines = []
    if active.numel() == 0:
        return ""
    for speaker in range(active.shape[1]):
        column = active[:, speaker].tolist()
        start = None
        for frame, on in enumerate(column + [False]):
            if on and start is None:
                start = frame
            elif not on and start is not None:
                onset = start * hop
                duration = (frame - start) * hop
                lines.append(
                    f"SPEAKER {uri} 1 {onset:.3f} {duration:.3f} <NA> <NA> spk{speaker} <NA> <NA>"
                )
                start = None
    return ("\n".join(lines) + "\n") if lines else ""


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

        pipe = SepRQPipeline("separation", upstream="SepRQ", streams=2)

        # Input mixture - [num_samples]
        mixture = "mixture.wav"

        save_to = "./output"

        sources = pipe(mixture, save_to=save_to)
        # [num_speakers, num_samples]

        # ./output/source1.wav
        # ./output/source2.wav

        pipe = SepRQPipeline("diarization", upstream="HuBERT_BASE")

        # Input mixture - [num_samples]
        mixture = "mixture.wav"

        save_to = "./output"

        activations = pipe(mixture, save_to=save_to)
        # [time, num_speakers]

        # ./output/activations.rttm

        pipe = SepRQPipeline("enhancement", upstream="WavLM_BASE")

        # Input mixture - [num_samples]
        mixture = "noisy.wav"

        save_to = "./output"

        clean = pipe(mixture, save_to=save_to)
        # [num_samples]

        # ./output/cleaned.wav

        pipe = SepRQPipeline("target-speaker-extraction",
                             upstream="SepRQ",
                             streams=2)

        # Input mixture - [num_samples]
        mixture = "mixture.wav"
        enrollment = "enrollment.wav"

        save_to = "./output"

        target = pipe(mixture, enrollment=enrollment, save_to=save_to)
        # [num_samples]

        # ./output/extracted_audio.wav
    """

    def __init__(self, task, upstream="SepRQ", streams=2, repo_id=None, device=None):
        super().__init__()
        from seprq import REPO_ID, SepRQEncoder

        if task == "separation-2spk":
            task = "separation"
        if task not in TASKS:
            raise ValueError(
                f"task must be one of {list(TASKS)}, got {task!r}"
            )
        if upstream == "SepRQ" and streams != 2:
            raise ValueError(
                "Published heads are for SepRQ with streams=2. "
                f"Got streams={streams!r}."
            )
        if upstream not in SUPERB_UPSTREAMS:
            raise ValueError(
                f"No head is published for {upstream!r} on {task!r}. "
                f"Available upstreams: {list(SUPERB_UPSTREAMS)}."
            )

        self.task = task
        self.upstream_name = upstream
        self.repo_id = repo_id or REPO_ID
        self.encoder = SepRQEncoder(upstream, streams=streams, repo_id=self.repo_id, device=device)
        self.encoder.requires_grad_(False).eval()

        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            self.repo_id, f"superb/{_HUB_TASK.get(task, task)}/{upstream}/head.ckpt"
        )
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.n_states = int(blob["featurizer"].numel())
        self.register_buffer("layer_weights", blob["featurizer"].float())
        self._build_head(blob)
        self.to(self.encoder._device())
        self.eval()

    def _build_head(self, blob):
        if self.task in TARGET_TASKS:
            self.register_buffer("layer_weights_k", blob["featurizer_k"].float())
            self.register_buffer("layer_weights_v", blob["featurizer_v"].float())
            self.frame_shift = int(blob.get("frame_shift", 320))
            self.head = build_target_head(blob)
            return
        weights = blob["model"]
        if self.task == "diarization":
            self.frame_shift = int(blob["frame_shift"])
            self.num_speakers = int(blob["num_speakers"])
            self.head = _DiarizationHead(
                blob["input_dim"], blob["hidden_size"], self.num_speakers, blob["rnn_layers"]
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

    def _weighted(self, states, weights=None):
        if len(states) != self.n_states:
            raise RuntimeError(
                f"The {self.upstream_name} head expects {self.n_states} hidden states, "
                f"got {len(states)}."
            )
        layer_weights = self.layer_weights if weights is None else weights
        stacked = torch.stack([s.float() for s in states], dim=0)
        mixture = torch.softmax(layer_weights, dim=-1)
        return (mixture.view(-1, 1, 1, 1) * stacked).sum(dim=0)

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

    def _collate(self, audio, wav_lens):
        """Return ``[batch, num_samples]``, per-item lengths, and relative lengths."""
        batched = isinstance(audio, (list, tuple)) and audio and (
            _is_path(audio[0]) or torch.is_tensor(audio[0]) or isinstance(audio[0], np.ndarray)
        )
        if batched:
            if _is_path(audio[0]):
                waves = [self.encoder._load(item) for item in audio]
            else:
                waves = []
                for item in audio:
                    wave = torch.as_tensor(item, dtype=torch.float32).reshape(-1)
                    waves.append(wave)
            width = max(int(wave.numel()) for wave in waves)
            batch = waves[0].new_zeros(len(waves), width)
            for index, wave in enumerate(waves):
                batch[index, : wave.numel()] = wave
            batch = batch.to(self.encoder._device())
            lengths = [int(wave.numel()) for wave in waves]
        else:
            batch = self.encoder._as_batch(audio)
            lengths = [batch.shape[1]] * batch.shape[0]
        if wav_lens is not None:
            relative = torch.as_tensor(wav_lens, dtype=torch.float32).flatten()
            lengths = [max(1, int((relative[i] * batch.shape[1]).round())) for i in range(batch.shape[0])]
        else:
            relative = torch.tensor([length / batch.shape[1] for length in lengths], dtype=torch.float32)
        return batch, lengths, relative

    def _write(self, result, save_to):
        root = Path(save_to)
        root.mkdir(parents=True, exist_ok=True)
        if self.task == "diarization":
            if torch.is_tensor(result) and result.dim() == 2:
                items = [result]
            elif torch.is_tensor(result):
                items = [result[index] for index in range(result.shape[0])]
            else:
                items = result
            for index, prob in enumerate(items):
                (_item_dir(root, index, len(items)) / "activations.rttm").write_text(
                    _rttm(prob, self.frame_shift, str(index))
                )
            return
        if self.task == "target-speaker-asr":
            texts = [result] if isinstance(result, str) else list(result)
            for index, text in enumerate(texts):
                (_item_dir(root, index, len(texts)) / "extracted_transcript.txt").write_text(
                    text + "\n", encoding="utf-8"
                )
            return
        if self.task == "personalized-vad":
            if torch.is_tensor(result) and result.dim() == 1:
                items = [result]
            elif torch.is_tensor(result):
                items = [result[index] for index in range(result.shape[0])]
            else:
                items = result
            for index, prob in enumerate(items):
                (_item_dir(root, index, len(items)) / "activations.rttm").write_text(
                    _rttm(prob.unsqueeze(-1), self.frame_shift, str(index))
                )
            return
        if self.task in ("enhancement", "target-speaker-extraction", "personalized-extraction"):
            name = "cleaned.wav" if self.task == "enhancement" else "extracted_audio.wav"
            if torch.is_tensor(result) and result.dim() == 1:
                waves = [result]
            elif torch.is_tensor(result):
                waves = [result[index] for index in range(result.shape[0])]
            else:
                waves = result
            for index, wave in enumerate(waves):
                _write_wav(_item_dir(root, index, len(waves)) / name, wave)
            return
        # separation: [speakers, samples], [batch, speakers, samples], or a list of the first
        if torch.is_tensor(result) and result.dim() == 2:
            items = [result]
        elif torch.is_tensor(result):
            items = [result[index] for index in range(result.shape[0])]
        else:
            items = result
        for index, item in enumerate(items):
            folder = _item_dir(root, index, len(items))
            for speaker in range(item.shape[0]):
                _write_wav(folder / f"source{speaker + 1}.wav", item[speaker])

    def _utterances(self, audio, wav_lens):
        batch, lengths, _relative = self._collate(audio, wav_lens)
        return [batch[index, : lengths[index]] for index in range(batch.shape[0])]

    def _speaker_embedding(self, wave):
        states = self.encoder.hidden_states(wave)
        keys = self._weighted(states, self.layer_weights_k)
        values = self._weighted(states, self.layer_weights_v)
        return self.head.embed(keys, values, None)[0]

    def _forward_target(self, audio, enrollment, save_to, wav_lens):
        if enrollment is None:
            raise ValueError(f"{self.task} takes an enrollment utterance (enrollment=).")
        mixtures = self._utterances(audio, wav_lens)
        enrollments = self._utterances(enrollment, None)
        if len(enrollments) == 1 and len(mixtures) > 1:
            enrollments = enrollments * len(mixtures)
        if len(enrollments) != len(mixtures):
            raise ValueError(
                "Pass one enrollment, or one enrollment per mixture. "
                f"Got {len(mixtures)} mixtures and {len(enrollments)} enrollments."
            )
        outputs = []
        for mixture, enroll in zip(mixtures, enrollments):
            embed = self._speaker_embedding(enroll)
            feat = self._weighted(self.encoder.hidden_states(mixture))[0]
            if self.task in ("target-speaker-extraction", "personalized-extraction"):
                outputs.append(self.head(feat, mixture, embed))
            elif self.task == "personalized-vad":
                frames = torch.tensor([feat.shape[0]], device=feat.device)
                outputs.append(self.head(feat.unsqueeze(0), frames, embed.unsqueeze(0))[0])
            else:
                frames = torch.tensor([feat.shape[0]], device=feat.device)
                outputs.append(self.head(feat.unsqueeze(0), frames, embed.unsqueeze(0), self.head.symbols)[0])
        if self.task == "target-speaker-asr":
            result = outputs[0] if len(outputs) == 1 else outputs
        else:
            result = outputs[0] if len(outputs) == 1 else (
                torch.stack(outputs) if len({item.shape[0] for item in outputs}) == 1 else outputs
            )
        if save_to is not None:
            self._write(result, save_to)
        return result

    def forward(self, audio, save_to=None, wav_lens=None, enrollment=None):
        """Run the selected task.

        audio is one path, a list of paths, or a 16 kHz waveform shaped
        ``[num_samples]``, ``[batch, num_samples]`` or
        ``[batch, channel, num_samples]``. One utterance returns
        ``[num_speakers, num_samples]``, ``[num_samples]``,
        ``[time, num_speakers]``, ``[time]``, or a string. A batch keeps a
        leading batch dimension. Target-speaker tasks also take ``enrollment``,
        one utterance or one per mixture.
        ``save_to`` is a relative directory: waveforms are wav files,
        diarization and personalized VAD are RTTM files, and target-speaker
        ASR is a text file per utterance.
        """
        if self.task in TARGET_TASKS:
            return self._forward_target(audio, enrollment, save_to, wav_lens)
        batch, lengths, relative = self._collate(audio, wav_lens)
        if self.task == "diarization":
            states = self.encoder.hidden_states(batch, relative)
            logits = self.head(self._weighted(states))
            prob = torch.sigmoid(logits)
            trimmed = []
            for index, length in enumerate(lengths):
                frames = min(prob.shape[1], max(1, math.ceil(length / self.frame_shift)))
                trimmed.append(prob[index, :frames])
            result = trimmed[0] if len(trimmed) == 1 else (
                torch.stack(trimmed) if len({item.shape[0] for item in trimmed}) == 1 else trimmed
            )
        else:
            outs = [self._separate(batch[index, :lengths[index]]) for index in range(batch.shape[0])]
            if self.task == "enhancement":
                waves = [item[0] for item in outs]
                result = waves[0] if len(waves) == 1 else (
                    torch.stack(waves) if len({wave.shape[0] for wave in waves}) == 1 else waves
                )
            else:
                stacked = [torch.stack(item) for item in outs]
                result = stacked[0] if len(stacked) == 1 else (
                    torch.stack(stacked) if len({item.shape[1] for item in stacked}) == 1 else stacked
                )
        if save_to is not None:
            self._write(result, save_to)
        return result
