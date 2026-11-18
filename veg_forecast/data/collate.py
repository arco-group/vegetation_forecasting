import numpy as np
import torch
import torch.nn.functional as F


def _pad(seqs, value=0.0):
    length = max(s.shape[0] for s in seqs)
    return torch.stack([F.pad(s, (0, 0, 0, length - s.shape[0]), value=value) for s in seqs])


def _pad_mask(lengths):
    length = max(lengths)
    return torch.arange(length)[None, :] >= torch.tensor(lengths)[:, None]


def collate_batch(items):
    """Pad variable-length windows. Padded rows are flagged in *_pad_mask (and as missing in *_mask).

    Also returns `target_delta_days` (B, S): days between the last historical observation and each target, used to
    down-weight distant targets in the training loss.
    """
    batch = {
        "index": torch.tensor([it["index"] for it in items], dtype=torch.long),
        "history": _pad([it["history"] for it in items]),
        "history_mask": _pad([it["history_mask"] for it in items], value=True),
        "history_pad_mask": _pad_mask([it["history"].shape[0] for it in items]),
        "future": _pad([it["future"] for it in items]),
        "future_mask": _pad([it["future_mask"] for it in items], value=True),
        "future_pad_mask": _pad_mask([it["future"].shape[0] for it in items]),
        "future_target_positions": torch.stack([it["future_target_positions"] for it in items]),
        "target": torch.stack([it["target"] for it in items]),
        "target_mask": torch.stack([it["target_mask"] for it in items]),
    }
    delta = []
    for it in items:
        end = np.datetime64(it["history_timestamps"][-1])
        days = (np.asarray(it["target_timestamps"], dtype="datetime64[ns]") - end).astype("timedelta64[D]")
        delta.append(days.astype(np.float32))
    batch["target_delta_days"] = torch.tensor(np.stack(delta), dtype=torch.float32)
    return batch


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
