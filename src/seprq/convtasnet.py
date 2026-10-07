"""WSJ0-mix ConvTasNet: free filterbank, temporal convolution, ReLU masks.

The published heads were trained at 8 kHz. They upsample that waveform to
16 kHz for the filterbank and the upstream, then downsample the sources.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GlobLN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, features):
        dims = list(range(1, features.ndim))
        mean = features.mean(dim=dims, keepdim=True)
        var = features.var(dim=dims, keepdim=True, unbiased=False)
        normed = (features - mean) / torch.sqrt(var + 1e-8)
        return (self.gamma * normed.transpose(1, -1) + self.beta).transpose(1, -1)


class _ConvBlock(nn.Module):
    def __init__(self, channels, hidden, skip, kernel, padding, dilation):
        super().__init__()
        self.shared_block = nn.Sequential(
            nn.Conv1d(channels, hidden, 1),
            nn.PReLU(),
            _GlobLN(hidden),
            nn.Conv1d(hidden, hidden, kernel, padding=padding, dilation=dilation, groups=hidden),
            nn.PReLU(),
            _GlobLN(hidden),
        )
        self.res_conv = nn.Conv1d(hidden, channels, 1)
        self.skip_conv = nn.Conv1d(hidden, skip, 1)

    def forward(self, features):
        shared = self.shared_block(features)
        return self.res_conv(shared), self.skip_conv(shared)


class _TDConvNet(nn.Module):
    def __init__(self, in_channels, n_src, n_filters):
        super().__init__()
        self.n_src = n_src
        self.out_chan = n_filters
        self.bottleneck = nn.Sequential(_GlobLN(in_channels), nn.Conv1d(in_channels, 128, 1))
        blocks = []
        for _repeat in range(3):
            for block in range(8):
                blocks.append(
                    _ConvBlock(128, 512, 128, 3, padding=2**block, dilation=2**block)
                )
        self.TCN = nn.ModuleList(blocks)
        self.mask_net = nn.Sequential(nn.PReLU(), nn.Conv1d(128, n_src * n_filters, 1))

    def forward(self, features):
        batch, _, frames = features.shape
        hidden = self.bottleneck(features)
        skip = hidden.new_zeros(batch, 128, frames)
        for block in self.TCN:
            residual, block_skip = block(hidden)
            skip = skip + block_skip
            hidden = hidden + residual
        score = self.mask_net(skip).view(batch, self.n_src, self.out_chan, frames)
        return F.relu(score)


def _fit_length(wave, length):
    if wave.shape[-1] == length:
        return wave
    if wave.shape[-1] > length:
        return wave[..., :length]
    return F.pad(wave, (0, length - wave.shape[-1]))


class WSJSeparator(nn.Module):
    """Separate an 8 kHz mixture. ``ssl`` is ``[batch, time, dim]`` at 16 kHz."""

    def __init__(self, n_filters, n_src, ssl_dim):
        super().__init__()
        import torchaudio

        self.n_filters = n_filters
        self.n_src = n_src
        self.stride = 8
        self.encoder_fb = _Filterbank(n_filters, 16)
        self.decoder_fb = _Filterbank(n_filters, 16)
        self.masker = _TDConvNet(n_filters + ssl_dim, n_src, n_filters)
        self.upsampler = torchaudio.transforms.Resample(8000, 16000, resampling_method="sinc_interp_kaiser")
        self.downsampler = torchaudio.transforms.Resample(16000, 8000, resampling_method="sinc_interp_kaiser")

    def load(self, blob):
        self.encoder_fb._filters.data.copy_(blob["encoder_filters"])
        self.decoder_fb._filters.data.copy_(blob["decoder_filters"])
        self.masker.load_state_dict(blob["masker"])

    def forward(self, wav8, ssl):
        """wav8 ``[batch, samples]``, ssl ``[batch, frames, dim]``. Returns ``[batch, n_src, samples]``."""
        if wav8.dim() == 1:
            wav8 = wav8.unsqueeze(0)
        length = wav8.shape[-1]
        wide = self.upsampler(wav8)
        mixture = F.conv1d(wide.unsqueeze(1), self.encoder_fb._filters, stride=self.stride)
        features = ssl.transpose(1, 2)
        factor = mixture.shape[-1] // features.shape[-1]
        features = features.repeat_interleave(factor, dim=-1)
        features = _fit_length(features, mixture.shape[-1])
        masks = self.masker(torch.cat((features, mixture), dim=1))
        masked = masks * mixture.unsqueeze(1)
        flat = masked.reshape(-1, self.n_filters, masked.shape[-1])
        decoded = F.conv_transpose1d(flat, self.decoder_fb._filters, stride=self.stride)
        decoded = decoded.view(wav8.shape[0], self.n_src, -1)
        decoded = self.downsampler(decoded)
        return _fit_length(decoded, length)


class _Filterbank(nn.Module):
    def __init__(self, n_filters, kernel):
        super().__init__()
        self._filters = nn.Parameter(torch.empty(n_filters, 1, kernel))
