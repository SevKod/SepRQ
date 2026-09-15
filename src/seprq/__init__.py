"""seprq -- SepRQ / BEST-RQ (50 Hz) SSL speech feature extractors.

The CNN frontend + 12-layer Conformer encoder and the global norm stats are
downloaded from the Hugging Face Hub (SevKod/SepRQ).

    from seprq import SepRQEncoder
    speech_encoder = SepRQEncoder("SepRQ", scale="reduced")  # model: "SepRQ"/"BestRQ_50Hz"; scale: "reduced"/"full"
    layers = speech_encoder("audio.wav")        # list of 12 Conformer layers, each [1, T, 576]
"""
import torch
from hyperpyyaml import load_hyperpyyaml
from huggingface_hub import hf_hub_download

SAMPLE_RATE = 16000

__all__ = ["SepRQEncoder", "MODELS", "REPO_ID"]
__version__ = "0.1.2"

REPO_ID = "SevKod/SepRQ"

# model name -> scale -> (subfolder in the repo, normalize-checkpoint filename)
# `None` means that scale is not released yet (TBA).
MODELS = {
    "SepRQ": {
        "reduced": ("SepRQ/reduced_scale/2_streams", "normalize.ckpt"),
        "full": None,  # TBA
    },
    "BestRQ_50Hz": {
        "reduced": ("BestRQ_50Hz/reduced_scale", "normalize_running_stats.ckpt"),
        "full": None,  # TBA
    },
}


class SepRQEncoder(torch.nn.Module):
    def __init__(self, model="SepRQ", scale="reduced", repo_id=REPO_ID, device=None):
        super().__init__()
        if model not in MODELS:
            raise ValueError(f"model must be one of {list(MODELS)}, got {model!r}")
        if scale not in MODELS[model]:
            raise ValueError(
                f"scale must be one of {list(MODELS[model])}, got {scale!r}"
            )
        entry = MODELS[model][scale]
        if entry is None:
            raise ValueError(
                f"{model!r} scale={scale!r} is not released yet (TBA)."
            )
        subdir, norm_name = entry
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Download ONLY the files needed for this model.
        yaml_path = hf_hub_download(repo_id, "seprq_inference.yaml")
        model_ckpt = hf_hub_download(repo_id, f"{subdir}/model.ckpt")
        norm_ckpt = hf_hub_download(repo_id, f"{subdir}/{norm_name}")

        with open(yaml_path) as f:
            hp = load_hyperpyyaml(f)

        self.melspec = hp["compute_features"].to(self.device)
        self.cnn = hp["CNN"].to(self.device)
        self.wrapper = hp["wrapper"].to(self.device)      # 12-layer Conformer

        # model.ckpt is the state_dict of ModuleList([CNN, wrapper]) -> 0.*/1.*
        torch.nn.ModuleList([self.cnn, self.wrapper]).load_state_dict(
            torch.load(model_ckpt, map_location=self.device)
        )
        # Global norm stats: the checkpoint stores running_mean / running_var.
        norm = torch.load(norm_ckpt, map_location=self.device)
        self.running_mean = norm["running_mean"].to(self.device)
        self.running_std = torch.sqrt(norm["running_var"].to(self.device) + 1e-5)

        self.cnn.eval()
        self.wrapper.eval()

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

    @torch.no_grad()
    def forward(self, audio):
        """audio: path to an audio file, or a 1D 16 kHz waveform (array/tensor).

        Returns the 12 Conformer layer outputs as a list, each [1, T, 576].
        Call the instance directly: ``speech_encoder(audio)``."""
        if isinstance(audio, (str, bytes)) or hasattr(audio, "__fspath__"):
            audio = self._load(audio)
        wavs = torch.as_tensor(audio, dtype=torch.float32).reshape(1, -1)
        wavs = wavs.to(self.device)
        wav_lens = torch.tensor([1.0], device=self.device)
        feats = self.melspec(wavs)                        # Fbank [1, T, 80]
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

        return outs                                       # 12 tensors, each [1, T, 576]
