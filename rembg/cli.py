import argparse
import glob
import os
from distutils.util import strtobool
from typing import BinaryIO
import sys
from pathlib import Path

import filetype
from tqdm import tqdm
import onnxruntime as ort

from .bg import remove
from .detect import ort_session

sessions: dict[str, ort.InferenceSession] = {}


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-m",
        "--model",
        default="u2net",
        type=str,
        choices=["u2net", "u2netp", "u2net_human_seg"],
        help="The model name.",
    )

    ap.add_argument(
        "-a",
        "--alpha-matting",
        nargs="?",
        const=True,
        default=False,
        type=lambda x: bool(strtobool(x)),
        help="When true use alpha matting cutout.",
    )

    ap.add_argument(
        "-af",
        "--alpha-matting-foreground-threshold",
        default=240,
        type=int,
        help="The trimap foreground threshold.",
    )

    ap.add_argument(
        "-ab",
        "--alpha-matting-background-threshold",
        default=10,
        type=int,
        help="The trimap background threshold.",
    )

    ap.add_argument(
        "-ae",
        "--alpha-matting-erode-size",
        default=10,
        type=int,
        help="Size of element used for the erosion.",
    )

    ap.add_argument(
        "-az",
        "--alpha-matting-base-size",
        default=1000,
        type=int,
        help="The image base size.",
    )

    ap.add_argument(
        "-p",
        "--path",
        nargs=2,
        help="An input folder and an output folder.",
    )

    ap.add_argument(
        "input",
        nargs=(None if sys.stdin.isatty() else "?"),
        default=(None if sys.stdin.isatty() else sys.stdin.buffer),
        type=argparse.FileType("rb"),
        help="Path to the input image.",
    )

    ap.add_argument(
        "output",
        nargs=(None if sys.stdin.isatty() else "?"),
        default=(None if sys.stdin.isatty() else sys.stdout.buffer),
        type=argparse.FileType("wb"),
        help="Path to the output png image.",
    )

    args = ap.parse_args()
    session = sessions.setdefault(args.model, ort_session(args.model))

    if args.path:
        full_paths = [os.path.abspath(path) for path in args.path]

        input_paths = [full_paths[0]]
        output_path = full_paths[1]

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        input_files = set()

        for input_path in input_paths:
            if os.path.isfile(path):
                input_files.add(path)
            else:
                input_paths += set(glob.glob(input_path + "/*"))

        for input_file in tqdm(input_files):
            input_file_type = filetype.guess(input_file)

            if input_file_type is None:
                continue

            if input_file_type.mime.find("image") < 0:
                continue

            out_file = os.path.join(
                output_path, os.path.splitext(os.path.basename(input_file))[0] + ".png"
            )

            Path(out_file).write_bytes(
                remove(
                    Path(input_file).read_bytes(),
                    session=session,
                    alpha_matting=args.alpha_matting,
                    alpha_matting_foreground_threshold=args.alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold=args.alpha_matting_background_threshold,
                    alpha_matting_erode_structure_size=args.alpha_matting_erode_size,
                    alpha_matting_base_size=args.alpha_matting_base_size,
                )
            )

    else:
        args.output.write(
            remove(
                args.input.read(),
                session=session,
                alpha_matting=args.alpha_matting,
                alpha_matting_foreground_threshold=args.alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=args.alpha_matting_background_threshold,
                alpha_matting_erode_structure_size=args.alpha_matting_erode_size,
                alpha_matting_base_size=args.alpha_matting_base_size,
            )
        )


if __name__ == "__main__":
    main()
