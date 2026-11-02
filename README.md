# AudioGround

Code for **AudioGround: Temporal Grounding in Audio via Deterministic Boundary Supervision** (IEEE Access, 2026), the journal extension of the Interspeech 2026 paper.

The repository contains

- the data-generation code that rebuilds the AudioGround-IT instruction-tuning data from public AudioCaps clips (`data/`),
- the AudioGround model and its training and inference code (`code/audioground/`),
- the evaluation protocol, benchmark builders, baseline runners, and analysis scripts (`eval/`, `experiments/`).

Model checkpoints are not released.

## Layout

```
code/audioground/        AudioGround model (SALMONN + frame-level interpolation, timestamp-conditioned
                         window-level Q-Former, absolute time embedding), train.py, infer.py, tools/, tests/
code/scripts/            environment setup and backbone weight download
data/scripts/            AudioGround-IT construction (clip preparation, caption embeddings, pools, synthesis)
data/configs/            data variants (position prior, junction types, similarity threshold)
eval/agl_eval/           unified evaluation protocol (parser, metrics, clipping, trivial baselines)
eval/runners/            zero-shot baselines and Qwen2.5-Omni fine-tuning
eval/scripts/            benchmark builders, probes, figures
experiments/train/       launcher and the configuration of every reported training run
experiments/infer/       inference on all benchmarks
experiments/eval/        tables with bootstrap confidence intervals
results/                 default output directory of the table and figure scripts
```

All paths are resolved relative to the repository root (`AG_ROOT`, set automatically by `experiments/env.sh` and `code/audioground/audioground/config.py`). Run Python scripts from the repository root unless stated otherwise.

## Setup

```bash
bash code/scripts/setup_env.sh          # conda env `audioground` (SALMONN-7B dependencies)
bash code/scripts/download_weights.sh   # SALMONN-7B, Vicuna-7B v1.5, Whisper-large-v2, BEATs, BERT -> ckpt/
```

The data-generation scripts additionally need `pandas pyarrow pyloudnorm scipy soundfile pyyaml sentence-transformers`.

## 1. Build AudioGround-IT

The source clips are the AudioCaps training clips. Place the audio parquets of the Hugging Face dataset `OpenSound/AudioCaps` under `data/raw/opensound_audiocaps/` and the caption parquet of `d0rj/audiocaps` under `data/raw/d0rj/`, then run

```bash
VARIANTS="lead" bash data/scripts/rebuild_all.sh
```

which prepares loudness-normalized 16 kHz clips, embeds the captions (Sentence-BERT), builds the clip pools, and synthesizes the variant. The variants used in the paper are

| config (`data/configs/`) | description |
|---|---|
| `default` | construction of the conference version (22% of the answers start at 0.0 s) |
| `lead` | leading segment, no answer at 0.0 s (main construction of the article) |
| `lead04`, `lead06`, `lead08`, `lead09` | intermediate shares of answers at 0.0 s |
| `nogap_lead`, `crossfade_lead`, `overlap_lead`, `bgmix_lead`, `mixjunction_lead` | junction augmentation |
| `tau*_lead` | similarity threshold of the clip pairing |

See `data/VARIANTS.md` for the row format.

## 2. Train

```bash
source experiments/env.sh
CUDA_VISIBLE_DEVICES=0,1 bash experiments/train/launch.sh E1L_fi_ts_learned 0     # AudioGround, seed 0
```

The global batch size is 24 for any number of GPUs. Run names in `experiments/train/configs/`:
`E1L_fi_ts_learned` is AudioGround (frame-level interpolation, timestamp conditioning, learned absolute time embedding) on the `lead` data;
`E1L_*` and `E7L_salmonn_ft` are the component ablation and the fine-tuned SALMONN-7B;
`E1_*` and `E0D*`/`E0Da*` use the data with 22% and intermediate shares of answers at 0.0 s;
`E4La_*` are the junction-augmentation and data variants, `E2La_*` the window, query-token and β sensitivity runs,
`E5La_*` the multitask comparisons, and `E6La_*` the short-target objectives.

## 3. Evaluate

```bash
python eval/scripts/build_benchmarks.py           # Clotho-Moment, UnAV-100, TUT-Sound Events 2017 (Original and Clipped)
python eval/scripts/build_tacos.py                # TACOS test split
CUDA_VISIBLE_DEVICES=0,1 bash experiments/infer/infer_benchmarks.sh $AG_RUNS/E1L_fi_ts_learned/0 default
python experiments/eval/paper_tables.py           # tables with 95% bootstrap confidence intervals
```

Unparseable responses are scored as misses. Zero-shot baselines are run with `experiments/eval/run_baselines.sh`, which expects the official code of each baseline under `eval/third_party/`.

## Citation

```bibtex
@inproceedings{kim2026audioground,
  author    = {Mingi Kim and Minchol Kwon and Junyeong Kim},
  title     = {{AudioGround}: Fine-Grained Temporal Grounding in Audio via Deterministic Boundary Supervision},
  booktitle = {Proc. Interspeech},
  year      = {2026},
  pages     = {887--892},
  doi       = {10.21437/Interspeech.2026-3467}
}
```

## License

Apache License 2.0 (see `LICENSE`). Third-party code included in this repository keeps its own license (see `NOTICE`). AudioCaps, the benchmarks, and the pretrained backbones are subject to their own terms.
