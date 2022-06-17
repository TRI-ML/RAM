import argparse
import urllib.error
import urllib.request
from pathlib import Path

import subprocess

# ANNOTATIONS_TAR_GZ = 'https://github.com/TAO-Dataset/annotations/archive/v1.0.tar.gz'
# Temporary URL while in beta.
ANNOTATIONS_TAR_GZ = 'https://achal-public.s3.amazonaws.com/release-beta/annotations/annotations.tar.gz'


def banner_log(msg):
    banner = '#' * len(msg)
    print(f'\n{banner}\n{msg}\n{banner}')


def log_and_run(cmd, *args, **kwargs):
    print(f'Running command:\n{" ".join(cmd)}')
    subprocess.run(cmd, *args, **kwargs)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('tao_root', type=Path)
    parser.add_argument('--split',
                        required=True,
                        choices=['train', 'val', 'test'])

    args = parser.parse_args()

    assert args.tao_root.exists(), (
        f'TAO_ROOT does not exist at {args.tao_root}')

    annotations_dir = args.tao_root / 'annotations'
    if annotations_dir.exists():
        print(f'Annotations directory already exists; skipping.')
    else:
        annotations_compressed = args.tao_root / 'annotations.tar.gz'
        if not annotations_compressed.exists():
            banner_log('Downloading annotations')
            try:
                urllib.request.urlretrieve(ANNOTATIONS_TAR_GZ,
                                           annotations_compressed)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    print(f'Unable to download annotations.tar.gz. Please '
                          f'download it manually from\n'
                          f'{ANNOTATIONS_TAR_GZ}\n'
                          f'and save it to {args.tao_root}.')
                    return
                raise
        banner_log('Extracting annotations')
        log_and_run([
            'tar', 'xzvf',
            str(annotations_compressed), '-C',
            str(args.tao_root)
        ])
        (args.tao_root / 'annotations-1.0').rename(annotations_dir)

    banner_log("Extracting BDD, Charades, HACS, and YFCC frames")
    log_and_run([
        'python', 'scripts/download/extract_frames.py',
        str(args.tao_root), '--split', args.split
    ])

    banner_log("Downloading AVA videos")
    log_and_run([
        'python', 'scripts/download/download_ava.py',
        str(args.tao_root), '--split', args.split
    ])

    banner_log("Verifying TAO frames")
    log_and_run([
        'python', 'scripts/download/verify.py',
        str(args.tao_root), '--split', args.split
    ])


if __name__ == "__main__":
    main()
