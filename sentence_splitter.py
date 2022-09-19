#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import multiprocessing
import sys
from typing import Any, IO

import spacy
from tqdm.auto import tqdm


# From https://github.com/allenai/allennlp/blob/b2eb036/allennlp/commands/__init__.py#L27
class ArgumentParserWithDefaults(argparse.ArgumentParser):
    """Custom argument parser that will display the default value for an argument in the help message."""

    _action_defaults_to_ignore = {"help", "store_true", "store_false", "store_const"}

    @staticmethod
    def _is_empty_default(default: Any) -> bool:
        if default is None:
            return True
        if isinstance(default, (str, list, tuple, set)):
            return not bool(default)
        return False

    def add_argument(self, *args, **kwargs) -> None:
        # Add default value to the help message when the default is meaningful.
        default = kwargs.get("default")
        if kwargs.get(
                "action"
        ) not in self._action_defaults_to_ignore and not self._is_empty_default(default):
            description = kwargs.get("help", "")
            kwargs["help"] = f"{description} (default = {default})"
        super().add_argument(*args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults(description="Program to split documents into sentences. The input should be "
                                                    "a text file with one document per line. The output will be one "
                                                    "sentence per line, and documents will be separated by an empty "
                                                    "line.")

    parser.add_argument("path", metavar="FILE", nargs="?", default="-", help="input file or '-' to read from stdin")
    parser.add_argument("--encoding", default="utf-8")

    parser.add_argument("--spacy-model-name", default="en_core_web_sm")

    parser.add_argument("--buffer-size", type=int, help="to control how much of the file is loaded into memory at once")
    parser.add_argument("--n-process", type=int, default=-1, help="to control how many processes are used run the "
                                                                  "spaCy pipeline. Especially useful when running on a "
                                                                  "CPU. Set to -1 to use one process when running on a "
                                                                  "GPU or to use all available cores when running on a "
                                                                  "CPU.")
    parser.add_argument("--use-gpu", choices=["no", "prefer", "require"], default="no")

    return parser.parse_args()


def get_file_line_count(path: str, **kwargs) -> int | None:
    if path == "-":
        return None
    else:
        with open(path, **kwargs) as file:
            return sum(1 for _ in file)


# From https://stackoverflow.com/a/29824059/1165181
@contextlib.contextmanager
def smart_open(path: str, mode: str | None = "r", **kwargs) -> IO[Any]:
    if path == "-":
        file = sys.stdin if mode is None or mode == "" or "r" in mode else sys.stdout
    else:
        file = open(path, mode, **kwargs)
    try:
        yield file
    finally:
        if path != "-":
            file.close()


def main() -> None:
    args = parse_args()

    line_count = get_file_line_count(args.path, encoding=args.encoding)

    if args.use_gpu == "no":
        use_gpu = False
    elif args.use_gpu == "prefer":
        use_gpu = spacy.prefer_gpu()
    elif args.use_gpu == "require":
        use_gpu = spacy.require_gpu()
    else:
        raise ValueError(f"Unknown value for --use-gpu: {args.use_gpu}")

    n_process = (1 if use_gpu else multiprocessing.cpu_count()) if args.n_process == -1 else args.n_process

    nlp = spacy.load(args.spacy_model_name)

    with smart_open(args.path, encoding=args.encoding) as file:
        for doc in tqdm(nlp.pipe(file, batch_size=args.buffer_size, n_process=n_process), total=line_count):
            for sent in doc.sents:
                if sentence_text := sent.text.strip():
                    print(sentence_text)
            print()


if __name__ == "__main__":
    main()
