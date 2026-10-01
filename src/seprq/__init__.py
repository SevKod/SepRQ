"""seprq -- SSL speech feature extractors for cocktail-party / multi-talker speech.

SepRQ and BEST-RQ (50 Hz) weights are pulled from the Hugging Face Hub
(SevKod/SepRQ). Generic baselines (HuBERT, WavLM) are pulled from the torchaudio
pipelines (torch hub) -- nothing extra to host.

    from seprq import SepRQEncoder
    encoder = SepRQEncoder("SepRQ")        # SepRQ / BestRQ_50Hz / HuBERT_BASE / WavLM_BASE / WavLM_BASE_PLUS
    layers = encoder("audio.wav")           # list of the 12 Transformer-layer outputs
"""
import torch
from hyperpyyaml import load_hyperpyyaml
from huggingface_hub import hf_hub_download

SAMPLE_RATE = 16000

__all__ = ["SepRQEncoder", "MODELS", "TORCHAUDIO_BUNDLES", "REPO_ID"]
__version__ = "0.3.1"

REPO_ID = "SevKod/SepRQ"

# SepRQ / BEST-RQ models hosted on the Hub. 576-dim @ 50 Hz.
#   name -> (subfolder in the repo, normalization). The normalization is either a
#   normalize-checkpoint filename (global running mean/var) or "sentence" for
#   per-utterance normalization (no stats file needed).
MODELS = {
    "SepRQ": ("SepRQ/2_streams", "normalize.ckpt"),
    "BestRQ_50Hz": ("BestRQ_50Hz", "sentence"),
}

# SepRQ comes in variants trained to separate N speakers -> repo subfolder.
SEPRQ_STREAMS = {2: "SepRQ/2_streams", 3: "SepRQ/3_streams"}

# Generic SSL baselines fetched from torchaudio.pipelines (torch hub). 768-dim @ 50 Hz.
#   name -> torchaudio.pipelines bundle attribute.
TORCHAUDIO_BUNDLES = {
    "HuBERT_BASE": "HUBERT_BASE",
    "WavLM_BASE": "WAVLM_BASE",
    "WavLM_BASE_PLUS": "WAVLM_BASE_PLUS",
    "WavLM_LARGE": "WAVLM_LARGE",  # 24 layers, 1024-dim
}


class SepRQEncoder(torch.nn.Module):
    def __init__(self, model="SepRQ", streams=2, repo_id=REPO_ID, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model

        if model in MODELS:
            self.kind = "seprq"
            self._init_seprq(model, streams, repo_id)
        elif model in TORCHAUDIO_BUNDLES:
            self.kind = "torchaudio"
            self._init_torchaudio(model)
        else:
            raise ValueError(
                f"model must be one of "
                f"{list(MODELS) + list(TORCHAUDIO_BUNDLES)}, got {model!r}"
            )

        self.to(self.device)
        self.eval()  # default to eval; call .train() to fine-tune

    # ------------------------------------------------------------------ loaders
    def _init_seprq(self, model, streams, repo_id):
        if model == "SepRQ":
            if streams not in SEPRQ_STREAMS:
                raise ValueError(
                    f"streams must be one of {sorted(SEPRQ_STREAMS)}, got {streams!r}"
                )
            subdir, norm_spec = SEPRQ_STREAMS[streams], "normalize.ckpt"
        else:
            subdir, norm_spec = MODELS[model]  # BestRQ_50Hz (streams is ignored)
        # Download ONLY the files needed for this model.
        yaml_path = hf_hub_download(repo_id, "seprq_inference.yaml")
        model_ckpt = hf_hub_download(repo_id, f"{subdir}/model.ckpt")

        with open(yaml_path) as f:
            hp = load_hyperpyyaml(f)

        self.melspec = hp["compute_features"]
        self.cnn = hp["CNN"]
        self.wrapper = hp["wrapper"]                        # 12-layer Conformer

        # model.ckpt is the state_dict of ModuleList([CNN, wrapper]) -> 0.*/1.*
        torch.nn.ModuleList([self.cnn, self.wrapper]).load_state_dict(
            torch.load(model_ckpt, map_location="cpu")
        )

        # Normalization: either global running stats (from a ckpt) or sentence-level.
        if norm_spec == "sentence":
            from speechbrain.processing.features import InputNormalization

            self.norm_mode = "sentence"
            self.normalizer = InputNormalization(norm_type="sentence")
        else:
            self.norm_mode = "global"
            norm = torch.load(
                hf_hub_download(repo_id, f"{subdir}/{norm_spec}"), map_location="cpu"
            )
            # kept as buffers so they follow .to()/.cuda() and are saved in state_dict.
            self.register_buffer("running_mean", norm["running_mean"].float())
            self.register_buffer(
                "running_std", torch.sqrt(norm["running_var"].float() + 1e-5)
            )

    def _init_torchaudio(self, model):
        import torchaudio

        bundle = getattr(torchaudio.pipelines, TORCHAUDIO_BUNDLES[model])
        self.ssl = bundle.get_model()                       # downloaded from torch hub

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _load(path):
        """Load an audio file -> mono 16 kHz float32 waveform [samples]."""
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=True)  # [samp, ch]
        wav = torch.from_numpy(data).mean(1)                       # mono
        if sr != SAMPLE_RATE:
            import torchaudio

            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        return wav

    def _as_batch(self, audio):
        """Accept a path or a waveform of shape [N], [B, N] or [B, C, N]
        and return a float32 tensor [B, N] on the module's device."""
        if isinstance(audio, (str, bytes)) or hasattr(audio, "__fspath__"):
            audio = self._load(audio)
        x = torch.as_tensor(audio, dtype=torch.float32)
        if x.dim() == 1:                   # [num_samples] -> [1, num_samples]
            x = x.unsqueeze(0)
        elif x.dim() == 3:                 # [B, channel, num_samples] -> mono
            x = x.mean(1)
        elif x.dim() != 2:                 # expect [B, num_samples] otherwise
            raise ValueError(
                "audio must be a path or a tensor shaped [num_samples], "
                "[batch, num_samples] or [batch, channel, num_samples]"
            )
        return x.to(self._device())

    def _device(self):
        return next(self.parameters()).device

    # ------------------------------------------------------------------ forward
    def forward(self, audio, wav_lens=None):
        """Run the encoder and return the 12 Transformer layer outputs as a list,
        each ``[batch, T, D]`` (D = 576 for SepRQ/BEST-RQ, 768 for HuBERT/WavLM).

        audio : a path to an audio file, or a waveform tensor/array shaped
            ``[num_samples]``, ``[batch, num_samples]`` or
            ``[batch, channel, num_samples]`` (multi-channel is averaged to mono).
            Tensors are assumed to be 16 kHz.
        wav_lens : optional ``[batch]`` of relative lengths (1.0 = full length),
            used to build the padding mask for batches of uneven length.

        Gradients flow (no ``torch.no_grad``): put the module in ``train()`` mode
        to fine-tune it or plug it into a larger model."""
        x = self._as_batch(audio)
        device = x.device

        if self.kind == "torchaudio":
            lengths = None
            if wav_lens is not None:
                wav_lens = torch.as_tensor(wav_lens, dtype=torch.float32, device=device)
                lengths = (wav_lens * x.shape[1]).round().long()
            feats, _ = self.ssl.extract_features(x, lengths)   # list of layer outputs
            return feats                                       # [B, T, 768] x 12

        # --- SepRQ / BEST-RQ path ---
        if wav_lens is None:
            wav_lens = torch.ones(x.shape[0], device=device)
        else:
            wav_lens = torch.as_tensor(wav_lens, dtype=torch.float32, device=device)

        feats = self.melspec(x)                            # Fbank [B, T, 80]
        if self.norm_mode == "sentence":
            feats = self.normalizer(feats, wav_lens)       # per-utterance norm
        else:
            feats = (feats - self.running_mean) / self.running_std   # global norm

        outs = []

        def hook(_m, _i, o):
            outs.append(o[0] if isinstance(o, tuple) else o)

        handles = [layer.register_forward_hook(hook)
                   for layer in self.wrapper.transformer.encoder.layers]
        try:
            self.wrapper(self.cnn(feats), wav_lens)        # runs the 12 layers
        finally:
            for h in handles:
                h.remove()

        return outs                                        # [B, T, 576] x 12
