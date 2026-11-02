# AudioGround-IT variants

Root: `data`. Every variant has `train.jsonl` (48,900 rows) and `val.jsonl` (1,000 rows), plus `stats.json` and `config.resolved.yaml`, in `variants/<name>/`.

Row format: `{"audio": abs_wav_path, "instruction", "response", "task", "gt_intervals": [[s,e],...], "meta": {...}}`. The audio is 16 kHz mono int16 wav, stored content-addressed under `audio_store/<junction_mode>/`. Identical samples are shared across variants, so always use the `audio` path from the row.

| variant | train / val JSONL | audio | what differs from default |
|---|---|---|---|
| default | `variants/default/{train,val}.jsonl` | `audio_store/silence` | silence gap 0.3–0.5 s, tau 0.33 |
| crossfade | `variants/crossfade/{train,val}.jsonl` | `audio_store/crossfade` | same layout; equal-power crossfade 0.5–1.0 s (capped at 0.4× the shorter segment); GT boundary = crossfade midpoint |
| background-mix | `variants/bgmix/{train,val}.jsonl` | `audio_store/bgmix` | same layout and GT as default; continuous ambience bed (743 AudioCaps-train ambience clips, excluded from the foreground pool) at SNR U(10, 20) dB |
| no-gap | `variants/nogap/{train,val}.jsonl` | `audio_store/nogap` | same layout; hard cuts with no gap |
| strong-label tightened | `variants/strong/{train,val}.jsonl` | reuses `audio_store/silence` (identical audio to default) | GT edges tightened with AudioSet-Strong caption-matched events; 4,469 samples have changed GT |
| tau0.25 | `variants/tau0.25/{train,val}.jsonl` | `audio_store/silence` (shared where identical) | tau = 0.25, so fillers change |
| tau0.41 | `variants/tau0.41/{train,val}.jsonl` | `audio_store/silence` (shared where identical) | tau = 0.41, so fillers change |
| short-span oversampled | `variants/shortspan/{train,val}.jsonl` | `audio_store/silence` (O/F identical to default) | G/D span-bin weights [.35,.30,.20,.10,.05] over bins [1,3,5,10,20,30] s; default uses [.14,.14,.17,.20,.35] |
| conventional (E5) | `variants/conventional/{train,val}.jsonl` | reuses `audio_store/silence` (identical rows/ids/splits/order to default) | only `response` changes: grounding unchanged; duration = "It lasts about X seconds." (no interval); ordering = precedence only; frequency = count only ("It occurs N times."). `gt_intervals` kept; `meta.response_style="conventional"`, ordering rows have `meta.first_event` |

Overlap mode (`junction.mode: overlap`) is implemented (`configs/overlap.yaml`) but not generated.

Regenerate or add a variant:
`cd scripts && python synth.py --config ../configs/<name>.yaml [--override key=value ...]`
Then run `scripts/validate.py variants/<name>/train.jsonl 500`. The summary across all variants is in `variants/variants_summary.json`.
