"""seprq -- SepRQ / BEST-RQ (50 Hz) SSL speech feature extractors.

The CNN frontend + 12-layer Conformer encoder and the global norm stats are
downloaded from the Hugging Face Hub (SevKod/SepRQ).

    from seprq import SepRQEncoder
    speech_encoder = SepRQEncoder("SepRQ")     # or "BestRQ_50Hz"
    layers = speech_encoder("audio.wav")        # list of 12 Conformer layers, each [1, T, 576]
"""
import torch
from hyperpyyaml import load_hyperpyyaml
from huggingface_hub import hf_hub_download

SAMPLE_RATE = 16000

__all__ = ["SepRQEncoder", "MODELS", "REPO_ID"]
__version__ = "0.1.4"

REPO_ID = "SevKod/SepRQ"

# model name -> (subfolder in the repo, normalize-checkpoint filename)
MODELS = {
    "SepRQ": ("SepRQ/2_streams", "normalize.ckpt"),
    "BestRQ_50Hz": ("BestRQ_50Hz", "normalize_running_stats.ckpt"),
}


class SepRQEncoder(torch.nn.Module):
    def __init__(self, model="SepRQ", repo_id=REPO_ID, device=None):
        super().__init__()
        if model not in MODELS:
            raise ValueError(f"model must be one of {list(MODELS)}, got {model!r}")
        subdir, norm_name = MODELS[model]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Download ONLY the files needed for this model.
        yaml_path = hf_hub_download(repo_id, "seprq_inference.yaml")
        model_ckpt = hf_hub_download(repo_id, f"{subdir}/model.ckpt")
        norm_ckpt = hf_hub_download(repo_id, f"{subdir}/{norm_name}")

        with open(yaml_path) as f:
            hp = load_hyperpyyaml(f)

        self.melspec = hp["compute_features"]
        self.cnn = hp["CNN"]
        self.wrapper = hp["wrapper"]                       # 12-layer Conformer

        # model.ckpt is the state_dict of ModuleList([CNN, wrapper]) -> 0.*/1.*
        torch.nn.ModuleList([self.cnn, self.wrapper]).load_state_dict(
            torch.load(model_ckpt, map_location="cpu")
        )
        # Global norm stats, kept as buffers so they follow .to()/.cuda() and are
        # saved/restored with the module's state_dict.
        norm = torch.load(norm_ckpt, map_location="cpu")
        self.register_buffer("running_mean", norm["running_mean"].float())
        self.register_buffer(
            "running_std", torch.sqrt(norm["running_var"].float() + 1e-5)
        )

        self.to(self.device)
        self.eval()  # default to eval; call .train() to fine-tune

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

    def forward(self, audio, wav_lens=None):
        """Run the encoder and return the 12 Conformer layer outputs as a list,
        each ``[batch, T, 576]``.

        audio : a path to an audio file, or a waveform tensor/array shaped
            ``[num_samples]``, ``[batch, num_samples]`` or
            ``[batch, channel, num_samples]`` (multi-channel is averaged to mono).
            Tensors are assumed to be 16 kHz.
        wav_lens : optional ``[batch]`` of relative lengths (1.0 = full length),
            used to build the padding mask for batches of uneven length.

        Gradients flow (no ``torch.no_grad``): put the module in ``train()`` mode
        to fine-tune it or plug it into a larger model. Call the instance
        directly: ``speech_encoder(audio)``."""
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

        device = self.running_mean.device
        x = x.to(device)
        if wav_lens is None:
            wav_lens = torch.ones(x.shape[0], device=device)
        else:
            wav_lens = torch.as_tensor(wav_lens, dtype=torch.float32, device=device)

        feats = self.melspec(x)                           # Fbank [B, T, 80]
        feats = (feats - self.running_mean) / self.running_std   # global norm

        # capture each Conformer layer output via forward hooks
        outs = []

        def hook(_m, _i, o):
            outs.append(o[0] if isinstance(o, tuple) else o)

        handles = [layer.register_forward_hook(hook)
                   for layer in self.wrapper.transformer.encoder.layers]
        try:
            self.wrapper(self.cnn(feats), wav_lens)       # runs the 12 layers
        finally:
            for h in handles:
                h.remove()

        return outs                                       # 12 tensors [B, T, 576]
