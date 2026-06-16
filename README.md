# Scanned Photo Cropper

This folder includes `crop_scanned_photos.py`, a reusable script for cropping
individual printed photos out of flatbed scanner pages.

The script does not use an LLM. It uses image processing to:

- find image files in a folder,
- detect printed-photo regions against a bright white scanner bed,
- save each detected photo as a separate JPEG,
- preserve a small edge around each printed photo,
- create a `contact_sheet.jpg` preview with filenames underneath.

Original scan files are read only. They are not modified.

## Requirements

Install Python dependencies:

```bash
python3 -m pip install pillow numpy
```

## Basic Use

From this folder:

```bash
python3 crop_scanned_photos.py
```

That reads image files in the current folder and writes results to:

```text
cropped/
```

The contact sheet will be:

```text
cropped/contact_sheet.jpg
```

## Use Another Folder

```bash
python3 crop_scanned_photos.py /path/to/scans --output cropped
```

If `--output` is a relative path, it is created inside the input folder.

## Diagnostic Overlays

For a new batch, it is useful to generate red-box previews showing what the
script detected:

```bash
python3 crop_scanned_photos.py --diagnostics
```

Diagnostics are written to:

```text
cropped/diagnostics/
```

Review `cropped/contact_sheet.jpg` after each run. If the red boxes or crops look
too tight, increase the edge margin:

```bash
python3 crop_scanned_photos.py --edge-margin 30
```

JPEG crop quality defaults to 95. To change it:

```bash
python3 crop_scanned_photos.py --jpeg-quality 90
```

If you previously generated PNG crops and want to remove old generated crop
files before writing JPEGs:

```bash
python3 crop_scanned_photos.py --clean-output
```

If the script includes too much scanner bed or misses very pale photo borders,
adjust the scanner-bed threshold:

```bash
python3 crop_scanned_photos.py --bed-min-brightness 245 --bed-max-chroma 12
```

Lower `--bed-min-brightness` makes the script classify more light pixels as bed.
Higher `--bed-min-brightness` is more conservative and may keep more pale border.

## Notes

- The input folder should contain scanner pages, not already-cropped photos.
- The script searches only the top level of the input folder, not subfolders.
- Existing files with the same JPEG output names are overwritten.
- `--clean-output` removes generated crop files matching names like
  `*_01.jpg`, `*_01.jpeg`, or `*_01.png` in the output folder.
- It works best when photos are separated by visible white scanner bed.
