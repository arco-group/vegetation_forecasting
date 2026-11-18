import numpy as np
from torch.utils.data import Subset


def temporal_split(dataset, cutoff="2019-06-01"):
    """Internal train / validation split by time, with a purge.

    Training: windows whose last target date is before `cutoff`. Validation: windows whose first history date is on
    or after `cutoff`. Windows straddling the cutoff are dropped, so no validation day is seen in training and early
    stopping is selected on an unseen period (as the test sets).
    """
    cut = np.datetime64(cutoff)
    first_hist = np.array([np.datetime64(str(h[0])[:10]) for h in dataset.history_timestamps])
    last_tgt = np.array([np.datetime64(str(t[-1])[:10]) for t in dataset.target_timestamps])
    train_idx = np.flatnonzero(last_tgt < cut).tolist()
    val_idx = np.flatnonzero(first_hist >= cut).tolist()
    if not train_idx or not val_idx:
        raise ValueError(f"temporal split at {cutoff} leaves an empty set (train {len(train_idx)}, val {len(val_idx)})")
    return Subset(dataset, train_idx), Subset(dataset, val_idx), len(dataset) - len(train_idx) - len(val_idx)
