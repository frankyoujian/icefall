# Fully formatted Zipformer ASR with MVQ knowledge distillation

This branch recovers the implementation associated with Jian You, Xiangfeng Li,
and Erwan Zerhouni, **"Enhancing Fully Formatted End-to-End Speech Recognition
with Knowledge Distillation via Multi-Codebook Vector Quantization"**,
ASRU 2025, DOI: [10.1109/ASRU65441.2025.11434761](https://doi.org/10.1109/ASRU65441.2025.11434761).

The distillation method comes from Liyong Guo et al., **"Predicting Multi-Codebook
Vector Quantization Indexes for Knowledge Distillation"**, ICASSP 2023,
DOI: [10.1109/ICASSP49357.2023.10095268](https://doi.org/10.1109/ICASSP49357.2023.10095268).
The contribution recovered here integrates that method into Zipformer and
evaluates fully formatted transcription on LibriSpeech-PC.

## Recovery provenance and scope

Sources were inspected read-only on **2026-11-18** under
`/data1/jianyou/K2/icefall/egs/librispeech/ASR/`. The reconstruction starts at the
server's Git HEAD, **`cea0dbe7b1cd4d5b7512c7974e53034ef456dd70`** (2024-08-28),
rather than a recent upstream revision. Source mtimes and experiment logs span
2024 and 2025. They are provenance clues, not proof that the present working tree
or a particular checkpoint produced every published result.

The main changes comprise eleven source files: three Zipformer core files, two
codebook-extraction files, and six decoding/scoring adapters. Core training,
model, and encoder files preserve the recovered source. The curated decoder
excludes a later Whisper experiment that overwrote actual Zipformer hypotheses;
the CTC adapter retains vocabulary-derived dimensions instead of an unrelated
hardcoded 756-token experiment; pretrained decoding retains ordinary logging.
These recovery choices do not redesign the historical training method.

The [source manifest](paper1_recovery_manifest.json) and
[experiment summary](paper1_experiments.json) record the recovery evidence.
Data, model weights, HDF5 teacher labels, logs, PDFs, scratch
files, SSH helpers, and credentials are not part of the branch.

## What was inherited and what changed

The existing `pruned_transducer_stateless6` recipe already supplied HuBERT
inference, teacher-embedding extraction, MVQ training/storage, independent
codebook classification, and the auxiliary-loss approach. The quantizer and
`JointCodebookLoss` remain supplied by `multi_quantization`. No new quantization
algorithm is implemented here. Although the reference paper describes a
HuBERT-Large/Conformer experiment, this Git baseline already contains the
HuBERT-XL wrapper; its presence is not a new change attributable to this paper.

The upstream history includes the original distillation integration
[`c4ee2bc0`](https://github.com/k2-fsa/icefall/commit/c4ee2bc0af8a758a8ae049f72619a111fb0667e2),
the later HuBERT fix
[`05dfd5e6`](https://github.com/k2-fsa/icefall/commit/05dfd5e630d525dcc8828feba4d9daf6624af319),
and distillation documentation
[`142420b3`](https://github.com/k2-fsa/icefall/commit/142420b3afa7b07c95f733c2e72ee80078364a44).

Paths below are relative to `egs/librispeech/ASR/`.

| Files | Recovered role |
| --- | --- |
| `zipformer/zipformer.py` | Return the selected intermediate stack and final representation as a list; use stack index 3 for KD. |
| `zipformer/model.py` | Connect `JointCodebookLoss(..., is_joint=False)` to intermediate features; pair teacher frames to match student time resolution. |
| `zipformer/train.py` | Load `cut.codebook_indexes`, pad with `-100`, add weighted KD during training, and expose the KD switch/loss scale. |
| `pruned_transducer_stateless6/extract_codebook_index.py`, `vq_utils.py` | Add explicit original/destination manifest paths and attach teacher labels while retaining current, potentially formatted, supervision text; remap cut IDs when needed. |
| `zipformer/decode.py` | Adapt the encoder return value and calculate normalized WER, WER C, WER PC, PER, and punctuation/casing F1. |
| `zipformer/streaming_decode.py` | Match KD checkpoint construction and add formatted/normalized scoring for streaming decoding. |
| `zipformer/punct_er.py` | Reused NVIDIA PER scorer; its copyright and Apache-2.0 attribution are retained. |
| `zipformer/export.py`, `pretrained.py`, `ctc_decode.py` | Match checkpoint construction and select the final encoder representation where required. |

The paper targets retain case and `. , ?`, separate punctuation from words, and
collapse whitespace. The recovered `data/` examples follow this convention and
contain a 500-token Unigram vocabulary. A complete source script that converts
LibriSpeech into this formatted corpus was not recovered in this changeset.

The main loss is the existing scheduled simple/pruned RNN-T objective plus
`alpha * codebook_loss`; KD is omitted from validation. Optional CTC/AED scale
changes coexist in the source but those heads are disabled in the paper's
RNN-T configuration. The encoder returns `[middle, final]` when KD is enabled
and `[final]` otherwise. Inference uses the final output and does not evaluate
the KD head. KD checkpoints still contain its parameters; the head has not been
physically stripped from every artifact. The streaming encoder API continues
to return its ordinary tensor representation.

## Paper settings and recovered evidence

| Setting | Paper description / recovered implementation |
| --- | --- |
| Teacher | `hubert_xtralarge_ll60k_finetune_ls960.pt`; transformer layer 36 of 48. |
| Student | Medium Zipformer; stack depths `2,2,3,4,3,2`; fourth stack, cumulative layer 11, width 512. |
| Features / tokenizer | 80-bin fbank; 500-token SentencePiece Unigram with mixed case and `. , ?`. |
| KD scale | Default `0.1`; ablations `0.05` and `0.2`. |
| Training | Paper: four 32 GB V100 GPUs, 30 epochs; existing logs also include 50-epoch runs. |
| Decoding | Beam 4; paper describes averaging ten checkpoints; old launcher uses `--avg 21`. |
| External LM | Three-layer LSTM, 1024 units, same vocabulary; ten training epochs and final checkpoint; shallow fusion. |

Two discrepancies are important when using the recovered code:

1. **Teacher dimensions:** the paper states 512-dimensional teacher embeddings
   and 128x compression. The inherited HuBERT-XL wrapper explicitly uses
   `embedding_dim=1280`; 512 is the selected student-stack width. The paper's
   dimensional/compression statement is not confirmed by this implementation.
2. **Teacher codebooks:** the paper says 16 teacher codebooks, while every
   current training-manifest entry audited under `data/vq_fbank_layer36_cb16`
   declares an array whose last dimension is **8**:

   | Subset | Entries with shape `[T, 8]` |
   | --- | ---: |
   | train-clean-100 | 78,123 |
   | train-clean-360 | 286,212 |
   | train-other-500 | 404,037 |
   | Total | 768,372 |

   These are complete manifest metadata counts, not a full HDF5-content audit.
   The model pairs adjacent 50 Hz teacher frames into one 25 Hz student frame,
   giving **8 teacher indexes -> 16 classifier targets**. Its fixed
   `num_codebooks=16` therefore does not verify 16 teacher codebooks. Generating
   16 teacher indexes would yield 32 targets after pairing and does not match
   this fixed configuration. The directory name `cb16` is not evidence of the
   stored shape. Historical extraction provenance sufficient to resolve the
   paper's claim was not recovered. This branch preserves the working code and
   records the discrepancy instead of silently changing the target definition.

The inherited extractor uses the first 1,000 clean-100 cuts to fit its quantizer,
whose codebook size is 256. It asserts index values in `[0,255]`; actual storage
dtype depends on `multi_quantization`, with no explicit cast in the extractor.
One actual clean-100 HDF5 array was also checked and confirmed to have shape
`[689, 8]` and dtype `uint8`. The KD path has no
explicit teacher/student time-shift delta, unlike the reference paper's
streaming experiments. Its pairing helper truncates excess teacher frames but
does not pad a too-short sequence, so feature/label alignment matters.

## Configuration templates

Run from `egs/librispeech/ASR/` on Linux with a suitable icefall environment.
These commands are configuration templates, not verified commands for exact
published runs. Supply matching formatted manifests, tokenizer, features,
audio, checkpoints, and teacher-label HDF5 files first. Existing log environments
report Python 3.10, PyTorch `2.0.1+cu117`, k2 `1.24.3`, and Lhotse `1.26.0`.
Install the recipe dependencies, `multi_quantization`, and fairseq for teacher
extraction. The recovered model still imports/requires `multi_quantization`
even with KD disabled.

### Training and ablations

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
common=(--world-size 4 --num-epochs 30 --start-epoch 1
        --use-fp16 True --full-libri True --max-duration 1000
        --bpe-model data/lang_bpe_500/bpe.model --return-cuts True
        --use-transducer True --use-ctc False --use-attention-decoder False
        --spec-aug-time-warp-factor -1)

# Non-streaming KD; existing [T,8] teacher labels pair into 16 targets.
python zipformer/train.py "${common[@]}" \
  --manifest-dir data/vq_fbank_layer36_cb16 \
  --exp-dir zipformer/exp_paper1_kd \
  --enable-distillation True --codebook-loss-scale 0.1

# No-KD baseline with the same formatted supervision/tokenizer.
python zipformer/train.py "${common[@]}" \
  --manifest-dir data/fbank --exp-dir zipformer/exp_paper1_no_kd \
  --enable-distillation False

# Streaming KD: these chunk/context lists are recipe defaults,
# not a claim about an exact published streaming configuration.
python zipformer/train.py "${common[@]}" \
  --manifest-dir data/vq_fbank_layer36_cb16 \
  --exp-dir zipformer/exp_paper1_streaming_kd \
  --enable-distillation True --codebook-loss-scale 0.1 --causal True \
  --chunk-size '16,32,64,-1' --left-context-frames '64,128,256,-1'

# Alpha ablations; run after the preceding jobs rather than concurrently
# on the same GPUs. For streaming no-KD, combine the baseline with --causal True.
for alpha in 0.05 0.2; do
  python zipformer/train.py "${common[@]}" \
    --manifest-dir data/vq_fbank_layer36_cb16 \
    --exp-dir "zipformer/exp_paper1_alpha_${alpha}" \
    --enable-distillation True --codebook-loss-scale "$alpha"
done
```

Teacher extraction still uses `pruned_transducer_stateless6/extract_codebook_index.py`.
Pass `--fbank data/fbank` and an explicit `--vq-fbank` destination: their defaults
point at an experiment directory. Existing-label reuse needs the expected
`*-vq.jsonl.gz` manifests and referenced HDF5 files; `--use-extracted-codebook True`
does not download them. Some from-scratch extraction inputs remain hardcoded to
`./data/fbank`, so `--fbank` is not a complete arbitrary-input override. Check the
8-to-16 alignment above before generating labels. Do not overwrite working
manifests to experiment with the unresolved 16-teacher-codebook setting.

### Decoding and metrics

Set `--enable-distillation` to match the checkpoint, even for inference. Use
`False` for the no-KD baseline and the matching experiment directory. Match all
architecture/tokenizer options to the training checkpoint. The examples use
the decoder's default running-model averaging. `--use-averaged-model False`
instead averages consecutive epoch checkpoints; neither mode selects the ten
best checkpoints automatically.

```bash
python zipformer/decode.py \
  --exp-dir zipformer/exp_paper1_kd --epoch 30 --avg 10 \
  --bpe-model data/lang_bpe_500/bpe.model --lang-dir data/lang_bpe_500 \
  --manifest-dir data/fbank --enable-distillation True \
  --decoding-method modified_beam_search --beam-size 4 --wer-type pc

python zipformer/streaming_decode.py \
  --exp-dir zipformer/exp_paper1_streaming_kd --epoch 30 --avg 10 \
  --bpe-model data/lang_bpe_500/bpe.model --manifest-dir data/fbank \
  --enable-distillation True --causal True \
  --chunk-size 32 --left-context-frames 256 \
  --decoding-method modified_beam_search --beam-size 4 --with-pc True
```

Non-streaming `--wer-type traditional` removes punctuation and normalizes case;
`c` removes punctuation but preserves case; `pc` preserves both and also computes
PER/F1. Streaming uses `--with-pc True` for WER PC/PER or `False` for normalized
WER; it does not implement separate WER C/F1. Neither decoder has a `--res-dir`
flag. **Copy or move the output directory after each metric pass**: the selected
metric is absent from filenames and a later run can overwrite earlier outputs.

The scorer normalizes curly apostrophes, retains ASCII letters/digits/apostrophes,
and collapses whitespace for traditional WER. WER PC is space-delimited, so use
the punctuation spacing expected by the corpus. PER uses the reused scorer's
denominator `correct + deletion + insertion + substitution`, including
insertions. Its word alignment remains case-sensitive. Punctuation/casing F1
uses only pairs whose normalized texts are equal (zero normalized WER), with
title case (`str.istitle()`) as the positive casing class. It micro-aggregates
positive labels and skips unequal word-sequence lengths. Empty/malformed inputs
are not fully guarded in the historical scorer.

For external-LM decoding, use `modified_beam_search_lm_shallow_fusion` with
`--use-shallow-fusion True --lm-type rnn --lm-exp-dir <matched-lm-directory>`,
`--lm-vocab-size 500 --rnn-lm-num-layers 3`, and embedding/hidden dimensions 1024.
Select the verified LM checkpoint and tune `--lm-scale`. The inherited LM
trainer starts at epoch 0: ten epochs normally produce `epoch-9.pt`, so the old
launcher's `--lm-epoch 9` is consistent with ten training epochs, not inherently
an off-by-one error. Complete formatted LM-data preparation was not recovered.

## Archived development path and exclusions

`paper1_stateless6_prototype.patch` preserves seven precursor files: the old
distillation shell, stateless6 train/decode changes, and four added Zipformer
support files. It is historical development evidence, not the main recipe.
From the repository root, use
`git apply --check egs/librispeech/ASR/zipformer/paper1_stateless6_prototype.patch`
before optionally applying it in a separate exploratory checkout. Its launcher
mixes 20 epochs, one GPU, a 756-token tokenizer, and disabled KD; download steps
are commented, and its alternate argument parsing rejects some model options.
Duplicated scaling/subsampling code and shape-debug prints are retained there.

Excluded working-tree changes include commented tokenizer probes in `prepare.sh`,
the incomplete 756-token/ngram `prepare_lm.sh` experiment, shared scaling debug
imports, RNN-LM comment-only edits, scratch files, and the later Whisper helper
and hypothesis override. The old non-streaming `decode.sh` also passes the stale
`--with-pc` option; use `--wer-type` as above. No datasets or pretrained assets
are bundled, and this recovery does not establish the exact checkpoints behind
the published tables.

## Validation and limits

Recovery checks included syntax/static review and CPU model smoke tests for
no-KD, non-streaming KD, and causal KD. Intermediate/final shapes and lengths
were checked; paired 8-to-16 targets gave finite KD losses and input gradients.
Scoring smoke checks covered all three non-streaming WER modes, F1 selection
and casing behavior, and PER for perfect, deleted, and substituted punctuation.
The no-KD and non-streaming KD models had 65,549,011 and 71,976,147 parameters
respectively, illustrating that the unused KD head remains in KD checkpoints.
No full training, GPU decoding, deployment/export runtime, or published-score
reproduction was performed. Downstream callers outside the adapted entrypoints
must account for the encoder's list return value.
Original source whitespace was retained for provenance; `git diff --check`
still reports historical trailing whitespace in several recovered lines.
