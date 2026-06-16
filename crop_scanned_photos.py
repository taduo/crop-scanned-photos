#!/usr/bin/env python3
"""
Crop printed photos from flatbed scanner pages.

The detector assumes each input image is a scanner page with one or more printed
photos on a bright, low-chroma scanner bed. It finds connected non-bed regions,
adds a small safety margin so photo edges are preserved, and writes each crop as
a separate JPEG.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    print(
        "Missing dependency. Install with: python3 -m pip install pillow numpy",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop individual printed photos from scanner-bed image sheets."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=".",
        type=Path,
        help="Folder containing scan images. Defaults to the current folder.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="cropped",
        type=Path,
        help="Folder for cropped JPEG files. Defaults to ./cropped.",
    )
    parser.add_argument(
        "--small-width",
        default=1000,
        type=int,
        help="Working width for detection. Larger is slower but can be more precise.",
    )
    parser.add_argument(
        "--bed-min-brightness",
        default=238,
        type=int,
        help="Minimum RGB channel value used to classify scanner bed pixels.",
    )
    parser.add_argument(
        "--bed-max-chroma",
        default=18,
        type=int,
        help="Maximum max(RGB)-min(RGB) value used to classify scanner bed pixels.",
    )
    parser.add_argument(
        "--min-area-ratio",
        default=0.012,
        type=float,
        help="Ignore detected regions smaller than this fraction of the scan.",
    )
    parser.add_argument(
        "--edge-margin",
        default=18,
        type=int,
        help="Full-resolution safety margin, in pixels, added around each crop.",
    )
    parser.add_argument(
        "--jpeg-quality",
        default=95,
        type=int,
        help="JPEG quality for cropped photos, from 1 to 100. Defaults to 95.",
    )
    parser.add_argument(
        "--contact-sheet",
        default="contact_sheet.jpg",
        help="Contact sheet filename, written inside the output folder.",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Write red-box preview images to <output>/diagnostics.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing generated crop files matching *_NN.jpg/png/jpeg first.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print warnings and the final summary.",
    )
    return parser.parse_args()


def image_files(input_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def clean_generated_outputs(output_dir: Path) -> int:
    removed = 0
    for pattern in ("*_??.jpg", "*_??.jpeg", "*_??.png"):
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()
                removed += 1
    return removed


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    height, width = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.zeros_like(mask)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result |= padded[dy : dy + height, dx : dx + width]
    return result


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    height, width = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.ones_like(mask)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result &= padded[dy : dy + height, dx : dx + width]
    return result


def connected_components(
    mask: np.ndarray, min_area: int
) -> list[tuple[int, int, int, int, int]]:
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[tuple[int, int, int, int, int]] = []

    for y in range(height):
        xs = np.flatnonzero(mask[y] & ~seen[y])
        for x0 in xs:
            if seen[y, x0] or not mask[y, x0]:
                continue

            queue: deque[tuple[int, int]] = deque([(int(x0), y)])
            seen[y, x0] = True
            min_x = max_x = int(x0)
            min_y = max_y = y
            area = 0

            while queue:
                x, cy = queue.popleft()
                area += 1
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)

                for nx, ny in ((x + 1, cy), (x - 1, cy), (x, cy + 1), (x, cy - 1)):
                    if (
                        0 <= nx < width
                        and 0 <= ny < height
                        and mask[ny, nx]
                        and not seen[ny, nx]
                    ):
                        seen[ny, nx] = True
                        queue.append((nx, ny))

            if area >= min_area:
                components.append((min_x, min_y, max_x + 1, max_y + 1, area))

    return components


def split_overlapping_boxes(boxes: list[list[int]]) -> None:
    """Remove overlaps introduced by safety padding between neighboring prints."""
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a = boxes[i]
            b = boxes[j]

            y_overlap = min(a[3], b[3]) - max(a[1], b[1])
            min_height = min(a[3] - a[1], b[3] - b[1])
            if y_overlap > min_height * 0.55:
                if a[0] <= b[0] and a[2] > b[0]:
                    split = (a[2] + b[0]) // 2
                    a[2] = split
                    b[0] = split
                elif b[0] < a[0] and b[2] > a[0]:
                    split = (b[2] + a[0]) // 2
                    b[2] = split
                    a[0] = split

            x_overlap = min(a[2], b[2]) - max(a[0], b[0])
            min_width = min(a[2] - a[0], b[2] - b[0])
            if x_overlap > min_width * 0.55:
                if a[1] <= b[1] and a[3] > b[1]:
                    split = (a[3] + b[1]) // 2
                    a[3] = split
                    b[1] = split
                elif b[1] < a[1] and b[3] > a[1]:
                    split = (b[3] + a[1]) // 2
                    b[3] = split
                    a[1] = split


def detect_boxes(image: Image.Image, args: argparse.Namespace) -> list[tuple[int, int, int, int]]:
    width, height = image.size
    small_width = args.small_width
    small_height = round(height * small_width / width)
    small = image.resize((small_width, small_height), Image.Resampling.BILINEAR)
    arr = np.asarray(small.convert("RGB"))

    channel_max = arr.max(axis=2)
    channel_min = arr.min(axis=2)
    chroma = channel_max - channel_min

    scanner_bed = (channel_min > args.bed_min_brightness) & (
        chroma < args.bed_max_chroma
    )
    mask = ~scanner_bed
    mask[:3, :] = False
    mask[-3:, :] = False
    mask[:, :3] = False
    mask[:, -3:] = False

    # Bridge photo content into one component per print while leaving gaps between
    # separate prints mostly intact.
    processed = erode(dilate(mask, 8), 3)
    min_area = int(args.min_area_ratio * small_width * small_height)
    components = connected_components(processed, min_area=max(1, min_area))

    boxes: list[list[int]] = []
    for x1, y1, x2, y2, area in components:
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width < 120 or box_height < 120:
            continue
        if area < args.min_area_ratio * small_width * small_height:
            continue

        small_pad = 9
        x1 = max(0, x1 - small_pad)
        y1 = max(0, y1 - small_pad)
        x2 = min(small_width, x2 + small_pad)
        y2 = min(small_height, y2 + small_pad)

        fx1 = round(x1 * width / small_width)
        fy1 = round(y1 * height / small_height)
        fx2 = round(x2 * width / small_width)
        fy2 = round(y2 * height / small_height)

        margin = args.edge_margin
        boxes.append(
            [
                max(0, fx1 - margin),
                max(0, fy1 - margin),
                min(width, fx2 + margin),
                min(height, fy2 + margin),
            ]
        )

    boxes.sort(key=lambda box: (box[1] // 500, box[0]))
    split_overlapping_boxes(boxes)
    return [tuple(box) for box in boxes]


def crop_scan(
    scan_path: Path, output_dir: Path, args: argparse.Namespace
) -> list[Path]:
    image = Image.open(scan_path).convert("RGB")
    boxes = detect_boxes(image, args)
    crop_paths: list[Path] = []

    for index, box in enumerate(boxes, start=1):
        crop = image.crop(box)
        crop_path = output_dir / f"{scan_path.stem}_{index:02d}.jpg"
        crop.save(
            crop_path,
            quality=max(1, min(100, args.jpeg_quality)),
            subsampling=0,
            optimize=True,
        )
        crop_paths.append(crop_path)

    if args.diagnostics:
        diagnostics_dir = output_dir / "diagnostics"
        diagnostics_dir.mkdir(exist_ok=True)
        write_diagnostic(scan_path, image, boxes, diagnostics_dir)

    if not args.quiet:
        print(f"{scan_path.name}: {len(crop_paths)} crops")

    return crop_paths


def write_diagnostic(
    scan_path: Path,
    image: Image.Image,
    boxes: Iterable[tuple[int, int, int, int]],
    diagnostics_dir: Path,
) -> None:
    preview = image.copy()
    preview.thumbnail((700, 990), Image.Resampling.LANCZOS)
    scale_x = preview.width / image.width
    scale_y = preview.height / image.height
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()

    for index, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        rect = (
            round(x1 * scale_x),
            round(y1 * scale_y),
            round(x2 * scale_x),
            round(y2 * scale_y),
        )
        draw.rectangle(rect, outline=(255, 0, 0), width=3)
        draw.text((rect[0] + 4, rect[1] + 4), str(index), fill=(255, 0, 0), font=font)

    preview.save(diagnostics_dir / f"{scan_path.stem}_boxes.jpg", quality=92)


def write_contact_sheet(crop_paths: list[Path], output_path: Path) -> None:
    if not crop_paths:
        return

    font = ImageFont.load_default()
    thumb_width = 240
    label_height = 26
    gap = 14
    columns = 6
    cells: list[Image.Image] = []

    for crop_path in crop_paths:
        image = Image.open(crop_path).convert("RGB")
        image.thumbnail((thumb_width, thumb_width), Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (thumb_width, thumb_width + label_height), "white")
        cell.paste(image, ((thumb_width - image.width) // 2, (thumb_width - image.height) // 2))

        draw = ImageDraw.Draw(cell)
        label = crop_path.name
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        draw.text(
            (max(2, (thumb_width - text_width) // 2), thumb_width + 7),
            label,
            fill=(0, 0, 0),
            font=font,
        )
        cells.append(cell)

    rows = (len(cells) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (
            columns * thumb_width + (columns + 1) * gap,
            rows * (thumb_width + label_height) + (rows + 1) * gap,
        ),
        (230, 230, 230),
    )

    for index, cell in enumerate(cells):
        x = gap + (index % columns) * (thumb_width + gap)
        y = gap + (index // columns) * (thumb_width + label_height + gap)
        sheet.paste(cell, (x, y))

    sheet.save(output_path, quality=94)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output.expanduser()
    if not output_dir.is_absolute():
        output_dir = input_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        removed = clean_generated_outputs(output_dir)
        if not args.quiet:
            print(f"Removed {removed} existing generated crop files from {output_dir}")

    scans = image_files(input_dir)
    if not scans:
        print(f"No image files found in {input_dir}", file=sys.stderr)
        return 1

    crop_paths: list[Path] = []
    for scan_path in scans:
        crop_paths.extend(crop_scan(scan_path, output_dir, args))

    contact_sheet_path = output_dir / args.contact_sheet
    write_contact_sheet(crop_paths, contact_sheet_path)

    print(f"Wrote {len(crop_paths)} crops to {output_dir}")
    if crop_paths:
        print(f"Wrote contact sheet to {contact_sheet_path}")
    if args.diagnostics:
        print(f"Wrote diagnostics to {output_dir / 'diagnostics'}")
    print("Review the contact sheet to confirm no photo edges were cut off.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
