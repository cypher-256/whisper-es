#./src/utils/monkeypatch_pooling.py
import torch
import torch.nn.functional as F
import warnings

from pyannote.audio.models.blocks.pooling import StatsPool


def _patched_forward(self, sequences: torch.Tensor, weights=None) -> torch.Tensor:
    if weights is None:
        mean = sequences.mean(dim=-1)
        if sequences.size(-1) > 1:
            std = sequences.std(dim=-1, correction=1)
        else:
            std = torch.zeros_like(mean)
        return torch.cat([mean, std], dim=-1)

    if weights.dim() == 2:
        has_speaker_dimension = False
        weights = weights.unsqueeze(dim=1)
    else:
        has_speaker_dimension = True

    _, _, num_frames = sequences.size()
    _, num_speakers, num_weights = weights.size()
    if num_frames != num_weights:
        warnings.warn(
            f"Mismatch between frames ({num_frames}) and weights ({num_weights}) numbers."
        )
        weights = F.interpolate(weights, size=num_frames, mode="nearest")

    def _pool(seq, w):
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
        mean = (seq * w.unsqueeze(1)).sum(dim=-1)
        if seq.size(-1) > 1:
            std = (((seq - mean.unsqueeze(-1)) ** 2) * w.unsqueeze(1)).sum(dim=-1).sqrt()
        else:
            std = torch.zeros_like(mean)
        return torch.cat([mean, std], dim=-1)

    output = torch.stack(
        [_pool(sequences, weights[:, speaker, :]) for speaker in range(num_speakers)],
        dim=1,
    )

    if not has_speaker_dimension:
        return output.squeeze(dim=1)

    return output


# Aplicar monkeypatch
StatsPool.forward = _patched_forward
