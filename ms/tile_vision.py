"""Colour-independent symbol recognition for captured Minesweeper cells."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "tile_vision.pt"
SIZE = 24
FLAG_CLASS, MINE_CLASS, OTHER = 9, 10, 11
MIN_CONFIDENCE = 0.995


class TileNet(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(48 * 6 * 6, 64), nn.ReLU(),
            nn.Linear(64, OTHER + 1),
        )


def normalize_symbol(ink):
    """Scale the foreground into a common frame while preserving its aspect ratio."""
    out = np.zeros((SIZE, SIZE), np.uint8)
    points = cv2.findNonZero(np.uint8(ink >= 64))
    if points is None:
        return out
    x, y, w, h = cv2.boundingRect(points)
    scale = (SIZE - 4) / max(w, h)
    width, height = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(ink[y:y+h, x:x+w], (width, height), interpolation=cv2.INTER_AREA)
    x, y = (SIZE - width) // 2, (SIZE - height) // 2
    out[y:y+height, x:x+width] = small
    return out


def observe(interior):
    """Return surface colour and a normalised symbol, or None for a uniform face."""
    # A wide, bold number can occupy most of a small cell. Sample the outline
    # of the inset face before estimating contrast, including on dark backgrounds.
    rim = np.concatenate((interior[0], interior[-1], interior[1:-1, 0], interior[1:-1, -1]))
    background = np.median(rim, axis=0).astype(np.float32)
    distance = np.linalg.norm(interior.astype(np.float32) - background, axis=2)
    pad = max(1, round(min(distance.shape) * .06))
    core = interior[pad:-pad, pad:-pad].astype(np.float32)
    if core.size and np.ptp(core, axis=(0, 1)).max() <= 16:
        # A bevel can tint the outer rim. A shallow linear background gradient
        # can also span the entire face. A localised faint stroke must survive.
        centre = np.median(core, axis=(0, 1))
        if np.max(np.abs(centre - background)) <= 12:
            h, w = core.shape[:2]
            dx = np.median(core[:, -1] - core[:, 0], axis=0)
            dy = np.median(core[-1] - core[0], axis=0)
            plane = (np.linspace(-.5, .5, w, dtype=np.float32)[None, :, None] * dx
                     + np.linspace(-.5, .5, h, dtype=np.float32)[:, None, None] * dy)
            residual = core - plane
            if np.ptp(residual, axis=(0, 1)).max() <= 2.5:
                return centre, None
    contrast = float(np.percentile(distance, 98))
    if contrast < 12:
        # A faint marking has insufficient evidence to declare the tile blank.
        return background, np.zeros((SIZE, SIZE), np.uint8)
    ink = np.clip((distance / contrast - 0.12) * (255 / 0.88), 0, 255).astype(np.uint8)
    return background, normalize_symbol(ink)


@lru_cache(maxsize=1)
def load_classifier():
    torch.set_num_threads(1)
    net = TileNet()
    net.load_state_dict(torch.load(WEIGHTS, weights_only=True, map_location="cpu"))
    return net.eval()


@torch.inference_mode()
def classify(symbols):
    if not len(symbols):
        return []
    inputs = torch.from_numpy(np.stack(symbols)).unsqueeze(1).float().div_(255)
    confidence, labels = load_classifier()(inputs).softmax(dim=1).max(dim=1)
    return [label if score >= MIN_CONFIDENCE and 1 <= label < OTHER else OTHER
            for label, score in zip(labels.tolist(), confidence.tolist())]
