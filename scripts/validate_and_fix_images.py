#!/usr/bin/env python3
"""
Validate and fix images in dataset folders.
- Moves corrupted/unreadable images to `data/bad_images/<split>/<class>/`
- Converts GIFs to JPEG (first frame) and moves original GIFs to backups

Usage: python scripts/validate_and_fix_images.py
"""
import os
import shutil
from pathlib import Path
from PIL import Image
import argparse


def is_image_valid(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def process_image(path: Path, bad_dir: Path) -> (bool, str):
    """Try to validate/convert an image.
    Returns (success, message).
    """
    suffix = path.suffix.lower()
    MAX_SIDE = 3000  # maximum allowed longest side in pixels
    # Try to open the image to get its size; if unreadable, move to bad_dir and continue
    try:
        with Image.open(path) as im:
            w, h = im.size
    except Exception as e:
        bad_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(path), str(bad_dir / path.name))
        except Exception:
            pass
        return False, f'unidentified:{e}'

    # If image is very large, downscale it (backup original first)
    if w * h > (MAX_SIDE * MAX_SIDE):
        try:
            backup_dir = bad_dir / 'originals'
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_dir / path.name)
            with Image.open(path) as im:
                im = im.convert('RGB')
                scale = min(float(MAX_SIDE) / max(w, h), 1.0)
                if scale < 1.0:
                    new_w = max(1, int(w * scale))
                    new_h = max(1, int(h * scale))
                    im_resized = im.resize((new_w, new_h), resample=Image.LANCZOS)
                    im_resized.save(path, quality=95)
                    return True, f'resized->{new_w}x{new_h}'
        except Exception as e:
            bad_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(path), str(bad_dir / path.name))
            except Exception:
                pass
            return False, f'resize_failed:{e}'

    # Handle GIFs: try to convert first frame to JPG
    if suffix == '.gif':
        try:
            with Image.open(path) as im:
                im.seek(0)
                frame = im.convert('RGB')
                new_path = path.with_suffix('.jpg')
                # avoid overwriting
                if new_path.exists():
                    new_path = path.with_name(path.stem + '_conv.jpg')
                frame.save(new_path, 'JPEG', quality=95)
            # move original gif to backups
            backup_dir = bad_dir / 'originals'
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(backup_dir / path.name))
            return True, f'converted->{new_path.name}'
        except Exception as e:
            # conversion failed -> move to bad_dir
            bad_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(path), str(bad_dir / path.name))
            except Exception:
                pass
            return False, f'moved_bad_gif:{e}'

    # For other formats, try loading
    try:
        with Image.open(path) as im:
            im.load()
        return True, 'ok'
    except Exception as e:
        bad_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(path), str(bad_dir / path.name))
        except Exception:
            pass
        return False, f'moved_bad:{e}'


def main(root='data', splits=None):
    if splits is None:
        splits = ['train', 'val', 'test']

    root_path = Path(root)
    bad_root = Path('data') / 'bad_images'
    bad_root.mkdir(parents=True, exist_ok=True)

    summary = {'total': 0, 'ok': 0, 'converted': 0, 'moved': 0, 'errors': []}

    for split in splits:
        split_dir = root_path / split
        if not split_dir.exists():
            print(f"Skipping missing split: {split_dir}")
            continue

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            bad_dir = bad_root / split / class_name
            bad_dir.mkdir(parents=True, exist_ok=True)

            for img_path in sorted(class_dir.glob('*')):
                if img_path.is_dir():
                    continue
                summary['total'] += 1
                success, msg = process_image(img_path, bad_dir)
                if success:
                    if msg.startswith('converted'):
                        summary['converted'] += 1
                    else:
                        summary['ok'] += 1
                else:
                    summary['moved'] += 1
                    summary['errors'].append((str(img_path), msg))

    # Print summary
    print('\n=== Validation Summary ===')
    print(f"Total files scanned: {summary['total']}")
    print(f"Valid files: {summary['ok']}")
    print(f"Converted (GIF->JPG): {summary['converted']}")
    print(f"Moved to bad_images: {summary['moved']}")
    if summary['errors']:
        print('\nSample errors (up to 20):')
        for p, m in summary['errors'][:20]:
            print(f"  {p} -> {m}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data', help='Dataset root (default: data)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'], help='Splits to scan')
    args = parser.parse_args()
    main(root=args.root, splits=args.splits)
