"""Target-speaker heads: extraction, personalized VAD, and target-speaker ASR."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

TARGET_TASKS = (
    "target-speaker-extraction",
    "personalized-extraction",
    "personalized-vad",
    "target-speaker-asr",
)

_SPECIALS = ("<s>", "<pad>", "</s>", "<unk>")


class _MHFA(nn.Module):
    """Multi-head factorized attentive pooling over an enrollment utterance."""

    def __init__(self, inputs_dim, compression_dim=128, head_nb=8, outputs_dim=256):
        super().__init__()
        self.cmp_linear_k = nn.Linear(inputs_dim, compression_dim)
        self.cmp_linear_v = nn.Linear(inputs_dim, compression_dim)
        self.att_head = nn.Linear(compression_dim, head_nb)
        self.pooling_fc = nn.Linear(head_nb * compression_dim, outputs_dim)

    def forward(self, k, v, lengths=None):
        k = self.cmp_linear_k(k)
        v = self.cmp_linear_v(v)
        att = self.att_head(k)
        if lengths is not None:
            steps = torch.arange(att.shape[1], device=att.device)
            att = att.masked_fill((steps.unsqueeze(0) >= lengths.to(att.device).unsqueeze(1)).unsqueeze(-1), float("-inf"))
        weights = torch.softmax(att, dim=1).unsqueeze(-1)
        pooled = torch.sum(v.unsqueeze(-2) * weights, dim=1)
        return self.pooling_fc(pooled.reshape(pooled.shape[0], -1))


class _TSE(nn.Module):
    def __init__(self, input_dim, hidden_size, rnn_layers, dropout, n_filters, kernel):
        super().__init__()
        self.model = nn.Module()
        self.model.rnn1 = nn.LSTM(input_dim, hidden_size, 1, batch_first=True, bidirectional=True)
        self.model.rnn2 = nn.LSTM(
            hidden_size * 2, hidden_size, rnn_layers - 1,
            batch_first=True, dropout=dropout, bidirectional=True,
        )
        self.model.drops = nn.Dropout(dropout)
        self.model.linear = nn.ModuleList([nn.Linear(hidden_size * 2, n_filters)])
        self.spk_extractor = _MHFA(input_dim, outputs_dim=hidden_size * 2)
        self.encoder = nn.Module()
        self.encoder.filterbank = nn.Module()
        self.encoder.filterbank._filters = nn.Parameter(torch.empty(n_filters, 1, kernel))
        self.decoder = nn.Module()
        self.decoder.filterbank = nn.Module()
        self.decoder.filterbank._filters = nn.Parameter(torch.empty(n_filters, 1, kernel))
        self.stride = 320
        self.kernel = kernel

    def embed(self, k, v, lengths):
        return self.spk_extractor(k, v, lengths)

    def forward(self, feat, wave, embed):
        """feat [T, D], wave [N], embed [1, D]. Returns [N]."""
        filters = self.encoder.filterbank._filters
        encoded = F.conv1d(wave.view(1, 1, -1), filters, stride=self.stride)
        frames = encoded.shape[-1]
        if abs(feat.shape[0] - frames) >= 20:
            raise RuntimeError(
                f"Upstream frames ({feat.shape[0]}) and filterbank frames ({frames}) differ by 20 or more."
            )
        if feat.shape[0] > frames:
            feat = feat[:frames]
        elif feat.shape[0] < frames:
            feat = F.pad(feat, (0, 0, 0, frames - feat.shape[0]))
        packed = torch.nn.utils.rnn.pack_sequence([feat])
        x, _ = self.model.rnn1(packed)
        x, lengths = pad_packed_sequence(x, batch_first=True)
        x = x * embed.view(1, 1, -1)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.model.rnn2(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        mask = torch.sigmoid(self.model.linear[0](self.model.drops(x)))
        masked = (mask * encoded.transpose(1, 2)).transpose(1, 2)
        wave_hat = F.conv_transpose1d(masked, self.decoder.filterbank._filters, stride=self.stride)
        wave_hat = wave_hat.view(-1)
        if wave_hat.numel() < wave.numel():
            wave_hat = F.pad(wave_hat, (0, wave.numel() - wave_hat.numel()))
        wave_hat = wave_hat[: wave.numel()]
        peak = wave_hat.detach().abs().max().clamp(min=1e-8)
        return wave_hat / peak * 0.99


class _PVAD(nn.Module):
    def __init__(self, input_dim, hidden_size, rnn_layers, outputs_dim):
        super().__init__()
        lstm_hidden = hidden_size // 2
        self.model = nn.Module()
        self.model.rnn1 = nn.LSTM(input_dim, lstm_hidden, 1, batch_first=True, bidirectional=True)
        self.model.rnn2 = nn.LSTM(hidden_size, lstm_hidden, rnn_layers - 1, batch_first=True, bidirectional=True)
        self.model.fc = nn.Linear(hidden_size, hidden_size)
        self.model.linear = nn.Linear(hidden_size, 3)
        self.spk_extractor = _MHFA(input_dim, outputs_dim=outputs_dim)

    def embed(self, k, v, lengths):
        return self.spk_extractor(k, v, lengths)

    def forward(self, feat, lengths, embed):
        lengths = lengths.detach().cpu().long()
        packed = pack_padded_sequence(feat, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.model.rnn1(packed)
        output, out_len = pad_packed_sequence(output, batch_first=True)
        output = output * embed.unsqueeze(1)
        packed = pack_padded_sequence(output, out_len.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.model.rnn2(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = torch.tanh(self.model.fc(output))
        return torch.softmax(self.model.linear(output), dim=-1)[..., 0]


class _RNNLayer(nn.Module):
    def __init__(self, input_dim, hidden, dropout):
        super().__init__()
        self.layer = nn.LSTM(input_dim, hidden, bidirectional=True, num_layers=1, batch_first=True)
        self.dp = nn.Dropout(dropout)
        self.out_dim = hidden * 2

    def forward(self, x, lengths):
        if not self.training:
            self.layer.flatten_parameters()
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.layer(packed)
        output, lengths = pad_packed_sequence(output, batch_first=True)
        return self.dp(output), lengths


class _TSASR(nn.Module):
    def __init__(self, input_dim, project_dim, dims, dropouts, outputs_dim, n_symbols):
        super().__init__()
        self.projector = nn.Linear(input_dim, project_dim)
        self.model = nn.Module()
        layers = []
        size = project_dim
        for dim, drop in zip(dims, dropouts):
            layer = _RNNLayer(size, dim, drop)
            layers.append(layer)
            size = layer.out_dim
        self.model.rnns = nn.ModuleList(layers)
        self.model.linear = nn.Linear(size, n_symbols)
        self.spk_extractor = _MHFA(input_dim, outputs_dim=outputs_dim)

    def embed(self, k, v, lengths):
        return self.spk_extractor(k, v, lengths)

    def forward(self, feat, lengths, embed, symbols):
        lengths = lengths.detach().cpu().long()
        x = self.projector(feat) * embed.unsqueeze(1)
        for layer in self.model.rnns:
            x, lengths = layer(x, lengths)
        logits = self.model.linear(x)
        texts = []
        for index in range(logits.shape[0]):
            ids = logits[index, : int(lengths[index])].argmax(-1).tolist()
            chars = []
            previous = 0
            for token in ids:
                if token != previous and token != 0:
                    symbol = symbols[token]
                    if symbol not in _SPECIALS:
                        chars.append(" " if symbol == "|" else symbol)
                previous = token
            texts.append("".join(chars).strip())
        return texts


def build_target_head(blob):
    """Build the head described by a stripped checkpoint and load its weights."""
    task = blob["task"]
    state = blob["state"]
    dim = int(blob["input_dim"])
    if task in ("target-speaker-extraction", "personalized-extraction"):
        head = _TSE(dim, blob["hidden_size"], blob["rnn_layers"], blob["dropout"], blob["n_filters"], blob["kernel"])
        head.stride = int(blob["stride"])
    elif task == "personalized-vad":
        head = _PVAD(dim, blob["hidden_size"], blob["rnn_layers"], blob["embed_dim"])
    elif task == "target-speaker-asr":
        head = _TSASR(dim, blob["project_dim"], blob["dims"], blob["dropouts"], blob["embed_dim"], blob["n_symbols"])
        head.symbols = blob["symbols"]
    else:
        raise ValueError(f"unknown target-speaker task {task!r}")
    head.load_state_dict(state)
    return head
