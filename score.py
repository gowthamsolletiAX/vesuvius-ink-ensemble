"""Score an ink prediction TIFF on a segment's held-out validation region.

usage: python score.py <segment_dir> <prediction.tif> [more.tif ...]
Several TIFFs are averaged first (a simple ensemble).
Labels: aligned family, annotated at z=10 of 21 (README of ink_9um).
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
from vc_reader.reader import open_node

Image.MAX_IMAGE_PIXELS = None
LABEL_Z = 10  # ponytail: aligned family only; native9 uses z=14


def plane(path):
    root = open_node(str(path))
    arr = root["0"] if not hasattr(root, "shape") else root
    return np.asarray(arr[LABEL_Z]) > 0


def score(seg_dir, preds):
    seg_dir = Path(seg_dir)
    name = seg_dir.name
    ink = plane(seg_dir / f"{name}_inklabels.zarr")
    val = plane(seg_dir / f"{name}_validation_mask.zarr")
    p = np.mean([np.asarray(Image.open(t), dtype=np.float32) / 255 for t in preds], axis=0)
    roi = Path(str(preds[0]) + ".roi.json")
    if roi.exists():  # windowed prediction: crop labels to the same window
        r0, r1, c0, c1 = json.loads(roi.read_text())["roi_row_start_stop_col_start_stop"]
        ink, val = ink[r0:r1, c0:c1], val[r0:r1, c0:c1]
    assert p.shape == ink.shape, (p.shape, ink.shape)
    y, s = ink[val], p[val]
    assert y.any() and (~y).any(), "validation region needs both ink and no-ink"
    hit = s > 0.5
    bacc = 0.5 * (hit[y].mean() + (~hit[~y]).mean())
    return dict(segment=name, n=int(val.sum()), ink_frac=round(float(y.mean()), 4),
                auc=round(float(roc_auc_score(y, s)), 4),
                ap=round(float(average_precision_score(y, s)), 4),
                bacc=round(float(bacc), 4))


if __name__ == "__main__":
    print(score(sys.argv[1], sys.argv[2:]))
