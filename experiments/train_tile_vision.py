"""Train a CPU symbol classifier for screenshots, independent of the playing policy."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn

from ms.tile_vision import (FLAG_CLASS, MINE_CLASS, OTHER, MIN_CONFIDENCE, SIZE,
                            TileNet, WEIGHTS, normalize_symbol, observe)


def font_bank(validation):
    styles = ("verdana.ttf", "georgiab.ttf", "tahomabd.ttf") if validation else (
        "arial.ttf", "arialbd.ttf", "ariblk.ttf", "segoeui.ttf", "segoeuib.ttf",
        "consolab.ttf", "impact.ttf", "calibrib.ttf", "timesbd.ttf")
    fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    faces = [ImageFont.truetype(str(fonts / name), 42) for name in styles if (fonts / name).is_file()]
    bank = {n: [] for n in range(1, OTHER + 1)}
    if not validation:
        # Include raster lettering as well as outline fonts. Enlarge before
        # varying stroke width so that counters in small bitmap faces survive.
        from ms.perception import _FONT5X7
        for value, rows in _FONT5X7.items():
            face = np.uint8(np.array([list(row) for row in rows]) == "#") * 255
            bank[value].extend([cv2.resize(face, None, fx=5, fy=5,
                                           interpolation=cv2.INTER_NEAREST)] * 5)
        faces.append(ImageFont.load_default(size=42))
    for char in "12345678?X+AZ@%*/":
        label = int(char) if char.isdigit() else OTHER
        for font in faces:
            glyph = Image.new("L", (80, 80))
            ImageDraw.Draw(glyph).text((8, 4), char, font=font, fill=255)
            bank[label].append(np.asarray(glyph))
        for face in ((cv2.FONT_HERSHEY_COMPLEX_SMALL,) if validation else
                     (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX)):
            for weight in (1, 3, 5):
                glyph = np.zeros((80, 80), np.uint8)
                cv2.putText(glyph, char, (8, 61), face, 1.8, 255, weight, cv2.LINE_AA)
                bank[label].append(glyph)
    if not validation:
        # Independent, labelled classic screenshot. Ocean is reserved for regression.
        path = Path(__file__).resolve().parents[1] / "tests/fixtures/classic-stalled.png"
        pixels = np.asarray(Image.open(path).convert("RGB"))
        for label, row, col in ((1, 0, 3), (2, 0, 2), (3, 1, 1), (FLAG_CLASS, 6, 4)):
            tile = pixels[row*24+2:(row+1)*24-2, col*24+2:(col+1)*24-2]
            _, glyph = observe(tile)
            bank[label].extend([glyph] * (12 if label != FLAG_CLASS else 1))
    return bank


def draw_icon(label, rng):
    glyph = np.zeros((48, 48), np.uint8)
    if label == FLAG_CLASS:
        pole = int(rng.integers(12, 35))
        tip = 3 if pole > 24 else 45
        thickness = int(rng.integers(2, 6))
        cv2.line(glyph, (pole, 5), (pole, 42), 255, thickness)
        if rng.random() < .5:
            cloth = [(pole, 4), (tip, rng.integers(5, 22)), (pole, rng.integers(22, 29))]
        else:
            # Rectangular and waving cloth, including single-colour flags without a base.
            floor = int(rng.integers(18, 32))
            tilt = int(rng.integers(-3, 6))
            middle = (pole + tip) // 2
            wave = int(rng.integers(-3, 4))
            cloth = [(pole, 4), (middle, 4 + wave), (tip, 4 + tilt),
                     (tip, floor + tilt), (middle, floor + wave), (pole, floor)]
        cv2.fillPoly(glyph, [np.array(cloth, np.int32)], 255)
        if rng.random() < .65:
            cv2.rectangle(glyph, (pole-8, 39), (pole+8, 44), 255, -1)
    elif label == MINE_CLASS:
        cv2.circle(glyph, (24, 24), int(rng.integers(9, 16)), 255, -1)
        for angle in np.arange(0, np.pi, np.pi / int(rng.integers(3, 6))):
            dx, dy = np.rint(20 * np.array([np.cos(angle), np.sin(angle)])).astype(int)
            cv2.line(glyph, (24-dx, 24-dy), (24+dx, 24+dy), 255, int(rng.integers(1, 5)))
        cv2.circle(glyph, (20, 19), int(rng.integers(1, 4)), 0, -1)
    else:
        for _ in range(int(rng.integers(1, 6))):
            x, y, xx, yy = rng.integers(2, 46, 4).tolist()
            cv2.line(glyph, (x, y), (xx, yy), 255, int(rng.integers(2, 6)))
    return glyph


# Stroke paths in a 5 x 7 design space for square-ended display lettering.
# Render a new weight, width and slant on each draw; no prototype is used at inference.
_STROKES = {
    1: [[(1,2),(3,0),(3,6)], [(1,6),(5,6)]],
    2: [[(0,1),(1,0),(4,0),(5,1),(5,2),(0,6),(5,6)]],
    3: [[(0,0),(4,0),(5,1),(5,2),(4,3),(1,3)], [(4,3),(5,4),(5,5),(4,6),(0,6)]],
    4: [[(3,0),(0,3.7),(5.2,3.7)], [(4,0),(4,6)]],
    5: [[(5,0),(0,0),(0,3),(4,3),(5,4),(5,5),(4,6),(0,6)]],
    6: [[(4,0),(2,0),(0,2),(0,5),(1,6),(4,6),(5,5),(5,4),(4,3),(0,3)]],
    7: [[(0,0),(5,0),(2,6)]],
    8: [[(1,0),(4,0),(5,1),(5,2),(4,3),(1,3),(0,2),(0,1),(1,0)],
        [(1,3),(0,4),(0,5),(1,6),(4,6),(5,5),(5,4),(4,3)]],
}


def display_digit(label, rng):
    glyph = np.zeros((64, 64), np.uint8)
    for path in _STROKES[label]:
        points = np.array(path, dtype=float) * 7 + [11, 10]
        points += rng.uniform(-1.8, 1.8, points.shape)
        layer = np.zeros_like(glyph)
        cv2.polylines(layer, [points.round().astype(np.int32)], False, 255, 2, cv2.LINE_AA)
        ky, kx = rng.integers(3, 14, 2)
        np.maximum(glyph, cv2.dilate(layer, np.ones((ky, kx), np.uint8)), out=glyph)
    return glyph


def samples(count, seed, validation):
    rng = np.random.default_rng(seed)
    bank = font_bank(validation)
    labels = np.arange(count) % (OTHER + 1)
    rng.shuffle(labels)
    inputs = np.zeros((count, SIZE, SIZE), np.uint8)
    for i, label in enumerate(labels):
        if label == 0:
            continue
        if not validation and 1 <= label <= 8 and rng.random() < .25:
            glyph = display_digit(label, rng)
        elif bank[label] and (label <= 8 or rng.random() < .45):
            glyph = bank[label][rng.integers(len(bank[label]))]
        else:
            glyph = draw_icon(label, rng)
        x, y, w, h = cv2.boundingRect(np.uint8(glyph > 63))
        crop = np.pad(glyph[y:y+h, x:x+w], 2)
        if validation:
            if rng.random() < .55:
                k = int(rng.integers(2, 4))
                crop = cv2.dilate(crop, np.ones((k, k), np.uint8))
        elif rng.random() < .60:
            ky, kx = rng.integers(1, 5, 2)
            crop = cv2.dilate(crop, np.ones((ky, kx), np.uint8))
        if not validation and rng.random() < .30:
            pixels = int(rng.integers(7, 15))
            small = cv2.resize(crop, (max(3, round(pixels * crop.shape[1] / crop.shape[0])), pixels),
                               interpolation=cv2.INTER_AREA)
            crop = cv2.resize(np.uint8(small >= rng.integers(65, 155)) * 255,
                              (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
        height = int(rng.integers(9, 38))
        width = max(3, round(height * crop.shape[1] / crop.shape[0] * rng.uniform(.70, 1.42)))
        glyph = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
        if rng.random() < .45:
            glyph = cv2.GaussianBlur(glyph, (3, 3), rng.uniform(.35, .8))
        glyph = np.uint8(255 * (glyph / 255) ** rng.uniform(.65, 1.6))
        inputs[i] = normalize_symbol(glyph)
    return torch.from_numpy(inputs[:, None]).float().div_(255), torch.from_numpy(labels).long()


def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(6)
    x, y = samples(args.samples, args.seed, False)
    model = TileNet()
    opt = torch.optim.AdamW(model.parameters(), lr=.0015)
    for epoch in range(args.epochs):
        model.train()
        for ids in torch.randperm(len(y)).split(192):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x[ids]), y[ids])
            loss.backward()
            opt.step()
        print(f"epoch {epoch+1} loss {loss.item():.4f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    evaluate(model, args)


def evaluate(model, args):
    model.eval()
    # Separate from the validation seed used to choose the rejection threshold.
    x, y = samples(3600, args.seed + 2, True)
    with torch.inference_mode():
        scores, predicted = torch.cat([model(batch) for batch in x.split(192)]).softmax(1).max(1)
    accepted = (scores >= MIN_CONFIDENCE) & (predicted > 0) & (predicted < OTHER)
    report = dict(seed=args.seed, training_examples=args.samples, epochs=args.epochs,
                  evaluation_seed=args.seed + 2, evaluation_examples=len(y),
                  min_confidence=MIN_CONFIDENCE, classes={})
    for n in range(OTHER + 1):
        group = y == n
        report["classes"][str(n)] = dict(examples=int(group.sum()),
            accepted=int((group & accepted).sum()), wrong_accepted=int((group & accepted & (predicted != y)).sum()))
    report["weights_sha256"] = sha256(args.out.read_bytes()).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=412081)
    p.add_argument("--samples", type=int, default=60000)
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--out", type=Path, default=WEIGHTS)
    p.add_argument("--report", type=Path, default=Path("reports/tile_vision.json"))
    train(p.parse_args())
