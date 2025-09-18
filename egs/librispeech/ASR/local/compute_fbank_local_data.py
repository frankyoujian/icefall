#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
This file computes fbank features of the LibriSpeech dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import sentencepiece as spm
import torch
from filter_cuts import filter_cuts
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import get_executor, str2bool

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bpe-model",
        type=str,
        help="""Path to the bpe.model. If not None, we will remove short and
        long utterances before extracting features""",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="""Dataset parts to compute fbank. If None, we will use all""",
    )

    parser.add_argument(
        "--perturb-speed",
        type=str2bool,
        default=False,
        help="""Perturb speed with factor 0.9 and 1.1 on train subset.""",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=20,
        help="Number of dataloading workers used for reading the audio.",
    )

    parser.add_argument(
        "--batch-duration",
        type=float,
        default=600.0,
        help="The maximum number of audio seconds in a batch."
        "Determines batch size dynamically.",
    )

    parser.add_argument(
        "--num-splits",
        type=int,
        default=0,
        help="The number of splits of the XL subset",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Process pieces starting from this number (inclusive).",
    )

    parser.add_argument(
        "--stop",
        type=int,
        default=-1,
        help="Stop processing pieces until this number (exclusive).",
    )

    return parser.parse_args()


def compute_fbank_librispeech(
    bpe_model: Optional[str] = None,
    dataset: Optional[str] = None,
    perturb_speed: Optional[bool] = True,
):
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")
    num_jobs = min(15, os.cpu_count())
    num_mel_bins = 80

    if bpe_model:
        logging.info(f"Loading {bpe_model}")
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)

    if dataset is None:
        dataset_parts = (
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        )
    else:
        dataset_parts = dataset.split(" ", -1)

    prefix = "librispeech"
    suffix = "jsonl.gz"
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=prefix,
        suffix=suffix,
    )
    assert manifests is not None

    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        len(dataset_parts),
        list(manifests.keys()),
        dataset_parts,
    )

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    with get_executor() as ex:  # Initialize the executor only once.
        for partition, m in manifests.items():
            cuts_filename = f"{prefix}_cuts_{partition}.{suffix}"
            if (output_dir / cuts_filename).is_file():
                logging.info(f"{partition} already exists - skipping.")
                continue
            logging.info(f"Processing {partition}")
            cut_set = CutSet.from_manifests(
                recordings=m["recordings"],
                supervisions=m["supervisions"],
            )

            if "train" in partition:
                if bpe_model:
                    cut_set = filter_cuts(cut_set, sp)
                if perturb_speed:
                    logging.info(f"Doing speed perturb")
                    cut_set = (
                        cut_set
                        + cut_set.perturb_speed(0.9)
                        + cut_set.perturb_speed(1.1)
                    )
            cut_set = cut_set.compute_and_store_features(
                extractor=extractor,
                storage_path=f"{output_dir}/{prefix}_feats_{partition}",
                # when an executor is specified, make more partitions
                num_jobs=num_jobs if ex is None else 80,
                executor=ex,
                storage_type=LilcomChunkyWriter,
            )
            cut_set.to_file(output_dir / cuts_filename)

# from gigaspeech and libriheavy
def compute_fbank_librispeech_splits(args):
    num_splits = args.num_splits
    subset=args.dataset
    src_dir = f"data/manifests/librispeech_{subset}_split"
    src_dir = Path(src_dir)
    output_dir = f"data/fbank/librispeech_{subset}_split"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assert output_dir.exists(), f"{output_dir} does not exist!"

    num_digits = 8  # num_digits is fixed by lhotse split-lazy

    start = args.start
    stop = args.stop
    if stop < start:
        stop = num_splits

    stop = min(stop, num_splits)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    # extractor = KaldifeatFbank(KaldifeatFbankConfig(device=device))
    num_mel_bins = 80
    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))
    logging.info(f"device: {device}")

    for i in range(start, stop):
        idx = f"{i}".zfill(num_digits)
        logging.info(f"Processing {idx}/{num_splits}")

        cuts_path = output_dir / f"librispeech_cuts_{subset}.{idx}.jsonl.gz"
        if cuts_path.is_file():
            logging.info(f"{cuts_path} exists - skipping")
            continue

        raw_cuts_path = src_dir / f"librispeech_cuts_{subset}.{idx}.jsonl.gz"
        if not raw_cuts_path.is_file():
            logging.info(f"{raw_cuts_path} does not exist - skipping it")
            continue

        logging.info(f"Loading {raw_cuts_path}")
        cut_set = CutSet.from_file(raw_cuts_path)

        logging.info("Computing features")
        if (output_dir / f"librispeech_feats_{subset}_{idx}.lca").exists():
            logging.info(f"Removing {output_dir}/librispeech_feats_{subset}_{idx}.lca")
            os.remove(output_dir / f"librispeech_feats_{subset}_{idx}.lca")

        cut_set = cut_set.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=f"{output_dir}/librispeech_feats_{subset}_{idx}",
            num_workers=args.num_workers,
            batch_duration=args.batch_duration,
            overwrite=True,
        )

        logging.info("About to split cuts into smaller chunks.")
        cut_set = cut_set.trim_to_supervisions(
            keep_overlapping=False, min_duration=None
        )

        logging.info(f"Saving to {cuts_path}")
        cut_set.to_file(cuts_path)
        logging.info(f"Saved to {cuts_path}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(vars(args))

    if args.num_splits:
        # assert args.num_splits is not None, "Please provide num_splits"
        compute_fbank_librispeech_splits(args)
    else:
        compute_fbank_librispeech(
            bpe_model=args.bpe_model,
            dataset=args.dataset,
            perturb_speed=args.perturb_speed,
        )
