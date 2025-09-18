#!/usr/bin/env python3
import logging
import re
import shutil
import tarfile
import zipfile
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import CutSet, fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import AlignmentItem, SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, is_module_available, resumable_download, safe_extract

import json
import sys

LIBRISPEECH = (
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
)
MINI_LIBRISPEECH = ("dev-clean-2", "train-clean-5")

def prepare_librispeech(
    corpus_dir: Pathlike,
    alignments_dir: Optional[Pathlike] = None,
    dataset_parts: Union[str, Sequence[str]] = "auto",
    output_dir: Optional[Pathlike] = None,
    normalize_text: str = "none",
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param alignments_dir: Pathlike, the path of the alignments dir. By default, it is
        the same as ``corpus_dir``.
    :param dataset_parts: string or sequence of strings representing dataset part names, e.g. 'train-clean-100', 'train-clean-5', 'dev-clean'.
        By default we will infer which parts are available in ``corpus_dir``.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param normalize_text: str, "none" or "lower",
        for "lower" the transcripts are converted to lower-case.
    :param num_jobs: int, number of parallel threads used for 'parse_utterance' calls.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    alignments_dir = Path(alignments_dir) if alignments_dir is not None else corpus_dir
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    # if dataset_parts == "mini_librispeech":
    #     dataset_parts = set(MINI_LIBRISPEECH).intersection(
    #         path.name for path in corpus_dir.glob("*")
    #     )
    # elif dataset_parts == "auto":
    #     dataset_parts = (
    #         set(LIBRISPEECH)
    #         .union(MINI_LIBRISPEECH)
    #         .intersection(path.name for path in corpus_dir.glob("*"))
    #     )
    #     if not dataset_parts:
    #         raise ValueError(
    #             f"Could not find any of librispeech or mini_librispeech splits in: {corpus_dir}"
    #         )
    # elif isinstance(dataset_parts, str):
    #     dataset_parts = [dataset_parts]
    if dataset_parts == "auto":
        dataset_parts = list(LIBRISPEECH)
    else:
        dataset_parts = dataset_parts.split()

    manifests = {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        manifests = read_manifests_if_cached(
            dataset_parts=dataset_parts, output_dir=output_dir
        )

    with ThreadPoolExecutor(num_jobs) as ex:
        for part in tqdm(dataset_parts, desc="Dataset parts"):
            logging.info(f"Processing LibriSpeech subset: {part}")
            if manifests_exist(part=part, output_dir=output_dir, prefix="librispeech"):
                logging.info(f"LibriSpeech subset: {part} already prepared - skipping.")
                continue
            recordings = []
            supervisions = []
            part_list = corpus_dir / part / "datalist.json"
            futures = []

            # for trans_path in tqdm(
            #     part_path.rglob("*.trans.txt"), desc="Distributing tasks", leave=False
            # ):
            #     alignments = {}
            #     ali_path = (
            #         alignments_dir
            #         / trans_path.parent.relative_to(corpus_dir)
            #         / (trans_path.stem.split(".")[0] + ".alignment.txt")
            #     )
            #     if ali_path.exists():
            #         alignments = parse_alignments(ali_path)
            #     # "trans_path" file contains lines like:
            #     #
            #     #   121-121726-0000 ALSO A POPULAR CONTRIVANCE
            #     #   121-121726-0001 HARANGUE THE TIRESOME PRODUCT OF A TIRELESS TONGUE
            #     #   121-121726-0002 ANGOR PAIN PAINFUL TO HEAR
            #     #
            #     # We will create a separate Recording and SupervisionSegment for those.
            #     with open(trans_path) as f:
            #         for line in f:
            #             futures.append(
            #                 ex.submit(parse_utterance, part_path, line, alignments)
            #             )
            with open(part_list, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    '''
                    item content example:
                        {
                            "id": "Fleurs0003",
                            "path": "/data1/jianyou/K2/Fleurs_70000/en/train_flac/14695089505774308.flac",
                            "text": "HELLO,WORLD",
                            "language": "English"
                        },
                    '''
                    futures.append(
                        ex.submit(parse_utterance, item)
                    )
            with RecordingSet.open_writer(
                output_dir / f"librispeech_recordings_{part}.jsonl.gz"
            ) as rec_writer, SupervisionSet.open_writer(
                output_dir / f"librispeech_supervisions_{part}.jsonl.gz"
            ) as sup_writer, CutSet.open_writer(
                output_dir / f"librispeech_cuts_{part}.jsonl.gz"
            ) as cut_writer:
                for future in tqdm(futures, desc="Processing", leave=False):
                    result = future.result()
                    if result is None:
                        continue
                    recording, segment = result
                    # recordings.append(recording)
                    # supervisions.append(segment)

                    # Fix and validate the recording + supervisions
                    recordings, segments = fix_manifests(
                        recordings=RecordingSet.from_recordings([recording]),
                        supervisions=SupervisionSet.from_segments([segment]),
                    )
                    validate_recordings_and_supervisions(
                        recordings=recordings, supervisions=segments
                    )
                    # Create the cut since most users will need it anyway.
                    # There will be exactly one cut since there's exactly one recording.
                    cuts = CutSet.from_manifests(
                        recordings=recordings, supervisions=segments
                    )
                    # Write the manifests
                    rec_writer.write(recordings[0])
                    sup_writer.write(segments[0])
                    cut_writer.write(cuts[0])
            manifests[part] = {
                "recordings": RecordingSet.from_jsonl_lazy(rec_writer.path),
                "supervisions": SupervisionSet.from_jsonl_lazy(sup_writer.path),
                "cuts": CutSet.from_jsonl_lazy(cut_writer.path),
            }

            # recording_set = RecordingSet.from_recordings(recordings)
            # supervision_set = SupervisionSet.from_segments(supervisions)

            # # Normalize text to lowercase
            # if normalize_text == "lower":
            #     to_lower = lambda text: text.lower()
            #     supervision_set = SupervisionSet.from_segments(
            #         [s.transform_text(to_lower) for s in supervision_set]
            #     )

            # recording_set, supervision_set = fix_manifests(
            #     recording_set, supervision_set
            # )
            # validate_recordings_and_supervisions(recording_set, supervision_set)

            # cuts = CutSet.from_manifests(
            #     recordings=recording_set, supervisions=supervision_set
            # )

            # if output_dir is not None:
            #     supervision_set.to_file(
            #         output_dir / f"librispeech_supervisions_{part}.jsonl.gz"
            #     )
            #     recording_set.to_file(
            #         output_dir / f"librispeech_recordings_{part}.jsonl.gz"
            #     )

            # manifests[part] = {
            #     "recordings": recording_set,
            #     "supervisions": supervision_set,
            # }

    return manifests


def parse_utterance(
    item: Dict[str, str],
    alignments: Dict[str, List[AlignmentItem]] = None,
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    # recording_id, text = line.strip().split(maxsplit=1)
    # # Create the Recording first
    # audio_path = (
    #     dataset_split_path
    #     / Path(recording_id.replace("-", "/")).parent
    #     / f"{recording_id}.flac"
    # )
    recording_id, audio_path, text, language = item['id'], Path(item['path']), item['text'], item['language']
    if not audio_path.is_file():
        logging.warning(f"No such file: {audio_path}")
        return None
    recording = Recording.from_file(audio_path, recording_id=recording_id)
    # Then, create the corresponding supervisions
    segment = SupervisionSegment(
        id=recording_id,
        recording_id=recording_id,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language=language,
        # speaker=re.sub(r"-.*", r"", recording.id),
        speaker=recording_id,
        text=text.strip(),
        # alignment={"word": alignments[recording_id]}
        # if recording_id in alignments
        # else None,
    )
    return recording, segment



download_dir = sys.argv[1]
output_dir = sys.argv[2]
data_parts = sys.argv[3]
num_jobs = 15

if len(sys.argv) > 4:
    num_jobs = int(sys.argv[4])

prepare_librispeech(corpus_dir=download_dir, dataset_parts=data_parts, output_dir=output_dir, num_jobs=num_jobs)