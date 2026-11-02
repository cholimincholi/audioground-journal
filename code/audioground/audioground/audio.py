"""Audio loading and long-audio chunk planning (CPU / dataloader side).

Timeline convention: Whisper encoder frame rate is 50 Hz (1500 frames per 30 s input).
Global frame f <-> time f / 50 s.  Every chunk is zero-padded to 30 s for Whisper and BEATs;
the model keeps `n_keep` frames of each chunk and places them at `start_frame` on the global timeline.
"""
import math
import numpy as np
import soundfile as sf

FPS = 50                 # Whisper encoder frames per second
WHISPER_SEC = 30.0       # Whisper fixed input length
WHISPER_FRAMES = 1500


def load_audio(path, sr=16000, max_sec=0.0):
    wav, file_sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1) if wav.shape[1] > 1 else wav[:, 0]
    if file_sr != sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr, res_type="fft").astype(np.float32)
    if max_sec and max_sec > 0:
        wav = wav[: int(max_sec * sr)]
    if len(wav) < sr:  # at least 1 s (as SALMONN)
        wav = np.concatenate([wav, np.zeros(sr - len(wav), dtype=np.float32)])
    return wav


def plan_chunks(duration_sec, chunk_sec=30.0, hop_sec=None, offset_sec=0.0, keep_final_padding=True):
    """Returns list of dicts {start_sec, end_sec, start_frame, n_keep}.

    - default (chunk = hop = 30, offset 0) reproduces the paper: non-overlapping 30 s chunks,
      ceil(D/30)*1500 frames when keep_final_padding.
    - offset_sec > 0 shifts the chunk grid: chunks start at 0, off, off+hop, ... with off = offset mod hop
      (e.g. offset 15 -> [0,15), [15,45), [45,75), ...).
    - hop_sec < chunk_sec gives overlapping chunks (frames merged in the model).
    Non-final chunks keep only their valid frames so frame index == time * 50 on the global timeline.
    """
    assert chunk_sec <= WHISPER_SEC + 1e-6, "Whisper cannot take chunks longer than 30 s"
    hop = chunk_sec if not hop_sec else float(hop_sec)
    assert 0 < hop <= chunk_sec
    off = float(offset_sec) % hop if offset_sec else 0.0
    starts = [0.0] if off > 1e-6 else []
    s = off
    while s < duration_sec - 1e-6:
        starts.append(s)
        s += hop
    chunks = []
    for i, st in enumerate(starts):
        end = min(st + chunk_sec, duration_sec)
        if hop >= chunk_sec and i + 1 < len(starts):
            end = min(end, starts[i + 1])
        is_last = i == len(starts) - 1
        n_valid = max(1, int(math.ceil((end - st) * FPS - 1e-6)))
        n_keep = int(round(chunk_sec * FPS)) if (is_last and keep_final_padding) else n_valid
        chunks.append({"start_sec": float(st), "end_sec": float(end),
                       "start_frame": int(round(st * FPS)), "n_keep": min(n_keep, WHISPER_FRAMES)})
    return chunks


def chunk_waveform(wav, chunks, sr=16000):
    """Slices + zero-pads every chunk to 30 s. Returns (N, 30*sr) float32 array and (N, 30*sr) bool pad mask."""
    n = int(WHISPER_SEC * sr)
    out = np.zeros((len(chunks), n), dtype=np.float32)
    pad = np.ones((len(chunks), n), dtype=bool)
    for i, c in enumerate(chunks):
        a, b = int(round(c["start_sec"] * sr)), int(round(c["end_sec"] * sr))
        seg = wav[a:b][:n]
        out[i, : len(seg)] = seg
        pad[i, : len(seg)] = False
    return out, pad


# ----------------------------------------------------------------------------- feature cache
# Frozen-encoder outputs (raw Whisper last_hidden_state (N,1500,1280) and BEATs features (N,~1496,768), pre-LayerNorm,
# exact bf16 bits stored as uint16) are cached per (audio file, chunking config). LN / interpolation / assembly stay
# online (LNs are trainable), so cached training is numerically identical to online encoding.
def chunk_tag(mcfg, sr=16000):
    hop = mcfg.chunk_hop_sec or mcfg.chunk_sec
    return (f"c{mcfg.chunk_sec:g}_h{hop:g}_o{mcfg.chunk_offset_sec or 0:g}_p{int(bool(mcfg.keep_final_padding))}"
            f"_m{mcfg.max_audio_sec or 0:g}_sr{sr}")


def cache_key(path, mcfg, sr=16000):
    import hashlib, os
    st = os.stat(path)
    s = f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}|{chunk_tag(mcfg, sr)}|whisper-large-v2|beats-iter3p-as2m-cpt2"
    return hashlib.sha1(s.encode()).hexdigest()


def cache_paths(root, path, mcfg, sr=16000):
    import os
    h = cache_key(path, mcfg, sr)
    d = os.path.join(root, chunk_tag(mcfg, sr), h[:2])
    return d, os.path.join(d, h)
