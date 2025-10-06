# Storage Diet

## 0) Keep pipeline, just drop ProRes HQ → ProRes 422 (standard)

* **Change:** `-c:v prores_ks -profile:v 2 -pix_fmt yuv422p10le` (profile 2 = ProRes 422).
* **Why:** Same 10-bit 4:2:2 mezz, ~25–55% smaller than HQ, visually transparent for SR input/output.
* **Upscaled mezz (4096×1728@25p):**

  * **5 min:** ~**10.8 GB** (was ~23.5 GB)
  * **128 min:** ~**~277 GB** (was ~600 GB)
* **Pre-mezz (1024×432@25p):**

  * **5 min:** ~**1.1 GB** (was ~1.47 GB)
  * **128 min:** ~**28 GB** (was ~37.5 GB)
* **Peak on-pod (mezz+final):** ~**320–340 GB** for full movie (vs ~660–670 GB)

**One-liner (writer):**

```bash
# replace HQ with profile 2
... -c:v prores_ks -profile:v 2 -pix_fmt yuv422p10le ...
```

---

## 1) ProRes LT for *pre-mezz only* (keep upscaled mezz at ProRes 422 std)

* **Change (pre-mezz):** `-profile:v 1` (ProRes LT).
* **Why:** The pre-mezz feeds SR; LT at 1024×432 is visually fine and halves that file without touching the main upscaled mezz quality.
* **Pre-mezz (1024×432@25p):**

  * **5 min:** ~**0.7–0.8 GB**
  * **128 min:** ~**18–20 GB**
* **Upscaled mezz:** keep as in option 0 (~10.8 GB / ~277 GB)

---

## 2) Skip writing pre-mezz to disk (stream it)

* **Change:** Feed the ffmpeg filtergraph (deinterlace→desqueeze→crop→709) **directly** into the Python runner via stdin or decord (no pre-mezz file).
* **Why:** Drops **~1.5 GB (sample)** and **~38 GB (full)** from your peak footprint with a tiny code change; the SR part stays identical.
* **Effect on peak:** subtract **~1.5 GB / ~38 GB** from the totals above.

**Pattern (decode stream → Python):**

```bash
ffmpeg -i "DVD_remux.mkv" -filter:v "bwdif=...,zscale=...,setsar=1,crop=1024:432:0:72" \
  -pix_fmt bgr24 -f rawvideo - | python tools/restoration_video_streaming.py --input - --raw-bgr ...
```

*(You already have the filters; this just avoids the intermediate file.)*

---

## 3) Use **HEVC All-Intra** mezzanine (NVENC), not ProRes

* **Change:** Pipe to `-c:v hevc_nvenc` with **all-intra** GOP (I-only), 10-bit, tuned for grain.
* **Why:** NVENC runs on a dedicated engine (doesn’t slow CUDA kernels), keeps quality high, and **shrinks mezz 3–6×** vs ProRes HQ.
* **Expected mezz size (4096×1728@25p, QP~18–20):**

  * **5 min:** **~4–8 GB**
  * **128 min:** **~100–200 GB**
* **Trade-off:** Slightly less “editing-friendly” than ProRes; visually very close if you choose conservative QP.

**NVENC all-intra example:**

```bash
ffmpeg -y -f rawvideo -pix_fmt bgr24 -s 4096x1728 -r 25 -i pipe:0 \
  -c:v hevc_nvenc -profile rext -pix_fmt p010le -preset p5 -tune hq \
  -rc constqp -qp 18 -g 1 -no-scenecut 1 \
  -color_primaries bt709 -color_trc bt709 -colorspace bt709 out_mezz_ai_hevc.mkv
```

---

## 4) **Direct-to-final x265** (no upscaled mezz file)

* **Change:** Pipe BasicVSR++ output straight to `libx265` (CRF 16, tune=grain); **do not** write the upscaled mezzanine.
* **Why:** Drops the **~600 GB** (or ~277 GB with option 0) file entirely.
* **Trade-off:** x265 is CPU-heavy; to keep pace with the GPU, use `-preset fast`/`faster`. Throughput likely **3–4 fps** instead of ~5 fps. Run time increases ~20–30%, but storage plummets.
* **Resulting sizes (same as your final):**

  * **5 min:** **~0.75–1.3 GB**
  * **128 min:** **~19–34 GB**
* **Peak on-pod:** basically **just the final** + small temp buffers (and the source). Huge win if time > storage.

**Direct pipe example:**

```bash
ffmpeg -y -f rawvideo -pix_fmt bgr24 -s 4096x1728 -r 25 -i pipe:0 \
  -vf "setpts=25/24*PTS,scale=-2:1440:flags=lanczos" -r 24 \
  -c:v libx265 -pix_fmt yuv420p10le -preset faster \
  -x265-params "crf=16:tune=grain:aq-mode=3:aq-strength=0.8:rc-grain=1:keyint=48:min-keyint=24" \
  out_final_1440p24.mkv
```

*(We integrate the 25→24 and resize here to truly skip the mezz.)*

---

## 5) Segment-then-delete

* **Change:** Keep your current writer, but **split the upscaled mezz** into e.g. **10-min chunks** (`output_%03d.mov`), kick off final x265 on each finished chunk, then delete that chunk.
* **Why:** **Reduces peak** (you never hold the whole mezz on disk at once) without changing codecs/tools.
* **Effect on peak:** ≈ size of **one chunk** + final (~**50–70 GB** if ProRes HQ chunks; **25–35 GB** if ProRes 422 std).

---

## What I recommend (balanced + minimal changes)

* **Quick wins (no perceptual hit):**

  1. **ProRes HQ → ProRes 422 (std)** for the **upscaled mezz** (Option 0).
  2. **ProRes LT** for the **pre-mezz** (Option 1).
  3. **Segment-then-delete** (Option 5).
     → Full movie peak drops from ~**660–670 GB** to roughly **320–360 GB** (or **~170–210 GB** if you also segment), with identical downstream workflow.

* **If storage is tight:**
  **NVENC HEVC All-Intra mezz** (Option 3) → Full mezz ~**100–200 GB**. Same overall flow; negligible GPU contention.

* **If storage is the priority over wall-clock:**
  **Direct-to-final x265** (Option 4) → Peak ~**20–35 GB**, but end-to-end time increases a bit (preset `fast/faster` to keep up).

---

## Side notes to preserve quality

* Keep everything **10-bit**, BT.709 flagged end-to-end.
* For NVENC all-intra, stay conservative (QP 18–20) and `-tune hq`.
* For ProRes, avoid 4:4:4 – 4:2:2 is enough; grain is luma-dominant.
* Do **subtitle OCR at 25 fps**, then retime to 24.000 (as discussed).

---

## Updated storage totals (at a glance)

**5-min sample**

* Current: **~26 GB peak**
* Option 0+1: **~12–14 GB**
* Option 3 (NVENC AI): **~6–10 GB**
* Option 4 (direct final): **~1–2 GB**

**Full 128-min**

* Current: **~660–670 GB peak**
* Option 0+1: **~320–360 GB**
* Option 3 (NVENC AI): **~130–230 GB**
* Option 5 (segment+delete): **~chunk size + final** (e.g., ~**50–70 GB** if 10-min ProRes 422 chunks)
* Option 4 (direct final): **~20–35 GB**
