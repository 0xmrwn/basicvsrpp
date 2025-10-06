# North Star: BasicVSR++ Upscaling on RunPod (reproducible, fast-enough, patchable)

- Target GPU pod: RunPod with CUDA 12.4, Python 3.11.
- Exact stack (as validated): PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, torchaudio 2.6.0+cu124, NumPy 2.2.6, SciPy 1.16.2, OpenCV-Python 4.12.0.88, mmcv-full 1.7.2 (compiled), mmedit 0.14.0 (from BasicVSR++ repo, editable). Two tiny patches applied to mmedit (see below).
- We run with a streaming inference script to avoid host-RAM explosion and improve throughput. We will keep that script in this repo and also add it (and a more advanced variant) to our fork of BasicVSR++ so future projects only require cloning two repos.

---

## Repos To Clone

- `movie-upscale` (this repo): holds orchestration docs, pre/post-processing commands, and a working streaming runner `restoration_video_streaming.py`.
- Your fork of BasicVSR++ (replace with your URL): `https://github.com/0xmrwn/BasicVSR_PlusPlus`.

---

## RunPod Pod Prep (one-time per pod)

- Base image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (or any CUDA 12.4 + Python 3.11 variant).
- Create work dirs: `mkdir -p /workspace/{models,tmp_pre,tmp_sr,out,sample}`.
- Optional: if `ffmpeg` is missing: `sudo apt-get update && sudo apt-get install -y ffmpeg`.

---

## Python Env + Core Wheels

```bash
# venv
python -m venv /opt/bsvrpp-venv
source /opt/bsvrpp-venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Torch 2.6.0 + CUDA 12.4 (as used in the working build)
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124

# NumPy 2.x + SciPy + OpenCV (NumPy 2.x is fine; we patch one alias)
pip install "numpy==2.2.6" "scipy==1.16.2" "opencv-python==4.12.0.88"
```

---

## Compile mmcv-full 1.7.2 (fits this Torch/CUDA)

```bash
pip install --no-binary mmcv-full "mmcv-full==1.7.2"

# Sanity
python - <<'PY'
import torch, mmcv
from mmcv.ops import deform_conv2d
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('MMCV:', mmcv.__version__, 'ops OK')
PY
```

---

## Clone Your Fork Of BasicVSR++ And Install (editable, no deps)

```bash
git clone https://github.com/<your-user>/BasicVSR_PlusPlus /opt/BasicVSR_PlusPlus
cd /opt/BasicVSR_PlusPlus
pip install --no-deps -e .
```

Apply two tiny patches in the fork so the modern stack works cleanly:

- Relax MMCV upper bound (accept 1.7.2): edit `mmedit/__init__.py` to change `MMCV_MAX = '1.6'` → `MMCV_MAX = '1.8'`.
- Remove deprecated NumPy alias: in `mmedit/datasets/pipelines/utils.py` comment out the `np.bool8` entry.

Commands (from `/opt/BasicVSR_PlusPlus`):

```bash
sed -i "s/MMCV_MAX = '1.6'/MMCV_MAX = '1.8'/" mmedit/__init__.py
sed -i "s/\s*np\.bool8: (False, True),/    # np.bool8 removed for NumPy 2.x/" mmedit/datasets/pipelines/utils.py
```

Commit these to your fork so every future clone is ready:

```bash
git checkout -b chore/env-pins-and-patches
git add mmedit/__init__.py mmedit/datasets/pipelines/utils.py
git commit -m "chore: relax mmcv<=1.6→<=1.8; drop np.bool8 for NumPy 2.x"
git push -u origin chore/env-pins-and-patches
# open PR and merge, or push directly to default branch
```

---

## Models/Checkpoints

```bash
mkdir -p /workspace/models && cd /workspace/models
curl -LO https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth
# Optional: rename to shorter
ln -s basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth basicvsr_plusplus_reds4.pth
```

---

## Preprocess (Mezzanine)

- For 25p interlaced DVD sample to 1024×432/25p 10‑bit mezzanine (ProRes 422 HQ). DNxHR HQX 10-bit is a quiet alternative if ProRes prints harmless packet-size warnings.

```bash
SAMPLE='Casanegra (2008) [DVD Remux] - SAMPLE.mkv'
OUT_PRE="/workspace/tmp_pre"; mkdir -p "$OUT_PRE"

ffmpeg -stats -y -i "/workspace/sample/$SAMPLE" \
 -filter:v "bwdif=mode=send_frame:parity=tff:deint=all,\
            zscale=w=1024:h=576:matrixin=bt470bg:matrix=bt709:filter=bicubic,\
            setsar=1,crop=1024:432:0:72" \
 -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
 -r 25 -c:v prores_ks -profile:v 3 -pix_fmt yuv422p10le \
 -map 0:v:0 -an -sn \
 "$OUT_PRE/main_SAMPLE_1024x432_25p_bt709.mov"
```

---

## Streaming Inference (baseline runner already in this repo)

- Script: `restoration_video_streaming.py` (writes PNGs to avoid RAM spikes).
- Example (25p mezzanine to x4):

```bash
source /opt/bsvrpp-venv/bin/activate
OUT_SR="/workspace/tmp_sr"; mkdir -p "$OUT_SR"

python /workspace/restoration_video_streaming.py \
  --config /opt/BasicVSR_PlusPlus/configs/basicvsr_plusplus_reds4.py \
  --checkpoint /workspace/models/basicvsr_plusplus_reds4.pth \
  --input /workspace/tmp_pre/main_SAMPLE_1024x432_25p_bt709.mov \
  --output "$OUT_SR" \
  --max-seq-len 250 \
  --device 0
```

Outputs PNGs to `/workspace/tmp_sr/%08d.png`.

---

## Integrate the Baseline Runner into Your Fork (mandatory)

We want the BasicVSR++ fork to be the long‑term home for streaming inference. First, copy today’s baseline as‑is (slow but functional) so we have a shared starting point for performance work.

```bash
cd /opt/BasicVSR_PlusPlus
git checkout -b feat/streaming-baseline
mkdir -p tools
cp /workspace/restoration_video_streaming.py tools/restoration_video_streaming.py
git add tools/restoration_video_streaming.py
git commit -m "feat(tools): add restoration_video_streaming.py baseline (PNG writer, FP32)"
git push -u origin feat/streaming-baseline
# open PR and merge; this becomes our baseline to iterate on
```

Run it from the fork (equivalent to the version in this repo):

```bash
python /opt/BasicVSR_PlusPlus/tools/restoration_video_streaming.py \
  --config /opt/BasicVSR_PlusPlus/configs/basicvsr_plusplus_reds4.py \
  --checkpoint /workspace/models/basicvsr_plusplus_reds4.pth \
  --input /workspace/tmp_pre/main_SAMPLE_1024x432_25p_bt709.mov \
  --output /workspace/tmp_sr \
  --max-seq-len 250 \
  --device 0
```

Why the baseline is abysmal for throughput (observed):

- Writes thousands of PNGs (CPU‑bound compression + heavy filesystem I/O).
- Random frame access per chunk (`vr[i]` pattern) can trigger frequent seeks depending on backend.
- Pure FP32 inference limits feasible `--max-seq-len` and underutilizes the 4090.
- No pinned memory or TF32 hints; little overlap between CPU decode and GPU compute.

---

## Enhance the Baseline In Your Fork (performance work happens here)

- In your `BasicVSR_PlusPlus` fork, evolve `tools/restoration_video_streaming.py` (the baseline you just copied) with:
  - mixed precision (AMP): wrap forward with `torch.cuda.amp.autocast(dtype=torch.float16)`;
  - TF32 enabled: `torch.backends.cuda.matmul.allow_tf32 = True`, `torch.backends.cudnn.allow_tf32 = True`;
  - pinned memory on host before H2D copy: `lq = lq.pin_memory()`;
  - sequential video reads into a rolling buffer to minimize random seeks;
  - optional `--writer ffmpeg` mode to pipe raw frames to ffmpeg for ProRes/FFV1 instead of writing thousands of PNGs.

Suggested CLI (inside `tools/restoration_video_streaming.py` after enhancements):

```text
python tools/restoration_video_streaming.py \
  --config .../basicvsr_plusplus_reds4.py \
  --checkpoint .../basicvsr_plusplus_reds4.pth \
  --input /path/in.mov \
  --output /path/out_dir_or_mov \
  --max-seq-len 96 \
  --writer ffmpeg \
  --device 0
```

Wire-up and commit in your fork:

```bash
cd /opt/BasicVSR_PlusPlus
git checkout -b feat/streaming-perf
# Edit tools/restoration_video_streaming.py to add AMP, TF32, pin_memory, sequential reads,
# and an ffmpeg writer (stdin rawvideo → ProRes/FFV1).
git add tools/restoration_video_streaming.py
git commit -m "perf(tools): AMP/TF32 + pinned mem + sequential reads + optional ffmpeg writer"
git push -u origin feat/streaming-perf
# open PR and merge
```

Runtime knobs for this faster path:

- Start with `--max-seq-len 96` and env `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- If VRAM OOM: drop to 80; if comfortable: try 120.
- If disk is slow: prefer `--writer ffmpeg` (ProRes/FFV1) over PNGs.

Expected improvement on a 4090 @ 1024×432 → x4: AMP+TF32 + ffmpeg pipe + sequential reads with T≈96 gives ~3–5× faster wall‑clock than FP32 + PNGs + random reads.

---

## Postprocess/Final Encode (to 1440p/24)

```bash
OUT_FIN="/workspace/out"; mkdir -p "$OUT_FIN"

# Pitch-preserved (time-stretch)
ffmpeg -stats -y -i "$OUT_SR/main_SAMPLE_basicvsrpp_x4.mp4" -i "/workspace/sample/$SAMPLE" \
 -filter:v "scale=-2:1440:flags=lanczos,setpts=25/24*PTS" -r 24 \
 -c:v libx265 -pix_fmt yuv420p10le -preset slow \
 -x265-params "crf=16:tune=grain:aq-mode=3:aq-strength=0.8:rc-grain=1" \
 -map 0:v:0 -map 1:a:0 -map_metadata 1 -map_chapters 1 -map 1:s? \
 -filter:a "atempo=24/25" -c:a ac3 -b:a 640k \
 "$OUT_FIN/main_SAMPLE_1440p24_pitchOK.mkv"

# Pitch-lowered alternative
ffmpeg -stats -y -i "$OUT_SR/main_SAMPLE_basicvsrpp_x4.mp4" -i "/workspace/sample/$SAMPLE" \
 -filter:v "scale=-2:1440:flags=lanczos,setpts=25/24*PTS" -r 24 \
 -c:v libx265 -pix_fmt yuv420p10le -preset slow \
 -x265-params "crf=16:tune=grain:aq-mode=3:aq-strength=0.8:rc-grain=1" \
 -map 0:v:0 -map 1:a:0 -map_metadata 1 -map_chapters 1 -map 1:s? \
 -af "asetrate=48000*24/25,aresample=48000" -c:a ac3 -b:a 640k \
 "$OUT_FIN/main_SAMPLE_1440p24_pitchLOW.mkv"
```

Quick QC helpers (optional): side‑by‑side preview; SSIM/PSNR vs mezzanine.

---

## What We Changed (vs. original attempts)

- Use upstream/forked BasicVSR++ repo directly; rely on its demo configs and weights.
- Build mmcv-full 1.7.2 to fit Torch 2.6/cu124; lift the repo’s `MMCV_MAX` gate to 1.8.
- Remove `np.bool8` to be NumPy 2.x‑safe.
- Switch from demo’s “collect all outputs” behavior to streaming inference to keep RAM flat.

---

## Performance Troubleshooting Checklist

- If GPU underutilized, check: are you writing PNGs (CPU/disk bound)? Use `--writer ffmpeg`.
- If VRAM OOM, reduce `--max-seq-len` or enable AMP+TF32.
- If frame reads are slow, use sequential buffering instead of random seeks.
- Set `torch.backends.cudnn.benchmark = True` when input size is stable.
- Consider `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for fewer alloc stalls.

---

## Areas To Research/Prototype Next (to further improve throughput)

- GPU decode path: evaluate feeding frames via NVDEC to reduce CPU decode cost (ffmpeg `-hwaccel cuda` pipe → Python) and measure end‑to‑end.
- `torch.compile` trial on the model’s forward (2.6): small A/B; if it compiles cleanly, can yield a few percent to double‑digit gains on long sequences.
- Mixed precision flavors: compare FP16 autocast vs BF16 (Ada can benefit; check accuracy and speed).
- Chunk overlap strategy: implement small temporal overlap (e.g., 8–12 frames) and keep only central region to avoid seams while preserving throughput.
- Tile‑over‑time trade‑offs: if VRAM limited at 4×, try spatial tiling with small overlap to sustain higher T.
- Alternative writer codecs: compare ProRes vs DNxHR HQX vs FFV1 for encode time and file size on your pod’s storage.
- mmcv version sweep: now that the fork lifts `MMCV_MAX`, A/B mmcv‑full 1.7.1 vs 1.7.2 vs 1.8.0 build to check tiny perf differences.
- Data prefetch threads and pinned memory: ensure pinned host tensors and overlap H2D copies; consider a simple producer/consumer to overlap decode and GPU compute.
- Multi‑GPU/distributed: not necessary, but for very long jobs consider splitting the timeline into contiguous segments per GPU with overlap to hide boundaries.

---

## One‑Command Recap For Next Project

1) Clone and prepare env:

```bash
git clone https://github.com/<your-user>/BasicVSR_PlusPlus /opt/BasicVSR_PlusPlus
git clone https://github.com/<your-user>/movie-upscale /workspace/movie-upscale
source /opt/bsvrpp-venv/bin/activate  # created as above
```

2) Ensure patches live in the fork (merged branch with MMCV_MAX/NumPy fix and `tools/restoration_video_streaming.py` baseline).

3) Run streaming inference via the tool in your fork or via `/workspace/movie-upscale/restoration_video_streaming.py` with the recommended flags.

This document is the single source of truth for pod prep, env versions, fork patches, and reproducible commands to go from source video → mezzanine → upscaled stream → final encode.
