# Optimization Plan for BasicVSR++ Video Upscaling

This document outlines a step-by-step plan to optimize the `tools/restoration_video_streaming.py` script for high-performance video upscaling using the BasicVSR++ model on an RTX 4090. The plan is based on the detailed analysis provided in the "Optimization Brief Executive Summary".

## Goal

The primary goal is to significantly improve the end-to-end throughput (frames per second) of the video restoration pipeline by addressing I/O bottlenecks, leveraging GPU hardware features, and optimizing memory usage. The target is to achieve a 3-5x performance increase over the baseline implementation.

---

### Step 1: Foundational Performance Enhancements

These changes provide the biggest performance boost by leveraging the RTX 4090's Tensor Cores and optimizing CUDA settings.

1.  **Enable Mixed Precision (AMP) and TF32**:
    *   Modify the inference loop to use `torch.cuda.amp.autocast(dtype=torch.float16)` to enable Automatic Mixed Precision. This will perform most computations in FP16, which is significantly faster on Tensor Cores.
    *   At the beginning of the script, enable TensorFloat-32 for any remaining FP32 operations:
        ```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

2.  **Enable cuDNN Benchmark**:
    *   For stable input sizes (which we have per chunk), enable cuDNN's auto-tuner to find the fastest convolution algorithms:
        ```python
torch.backends.cudnn.benchmark = True
```

### Step 2: I/O Pipeline Optimization

This step focuses on making the data flow from disk to GPU as efficient as possible.

1.  **Implement Sequential Frame Reading**:
    *   Modify the video reading loop to read frames sequentially from the `mmcv.VideoReader` object. The current implementation `vr[i]` can lead to slow, random-access seeks.
    *   A better approach is to iterate through the reader or use a method like `decord.VideoReader.get_batch()` if `decord` is available.

2.  **Use Pinned Memory and Asynchronous Transfers**:
    *   Before moving the input tensor (`lq`) to the GPU, pin the host memory using `.pin_memory()`.
    *   When transferring the tensor to the device, use `non_blocking=True`. This allows the CPU-to-GPU data transfer to overlap with GPU computation, hiding the data transfer latency.
    ```python
lq = frames_to_lq_tensor(frames, preprocess=preprocess).pin_memory()
...
out = model(lq=lq.to(device_str, non_blocking=True), test_mode=True)
```

### Step 3: High-Performance Output with FFmpeg

This is a critical optimization to remove the bottleneck of writing thousands of individual PNG files.

1.  **Add FFmpeg Writer Option**:
    *   Add a `--writer` command-line argument that can be set to `ffmpeg`.
    *   When `ffmpeg` is selected, the script will start an `ffmpeg` subprocess using `subprocess.Popen`.
    *   The output frames from the model will be piped directly to the `stdin` of the `ffmpeg` process.

2.  **Configure FFmpeg for a Mezzanine Codec**:
    *   The `ffmpeg` command should be configured to encode the raw video frames into a high-quality, fast-encoding intermediate (mezzanine) codec like **ProRes 422 HQ**.
    *   Example `ffmpeg` arguments for ProRes:
        ```bash
ffmpeg -y -f rawvideo -pix_fmt bgr24 -s {W}x{H} -r {fps} -i pipe:0 \
               -c:v prores_ks -profile:v 3 -pix_fmt yuv422p10le output.mov
```
    *   The script should handle the creation of this command and the writing of frame data to the pipe.

### Step 4: Seamless Tiling with Chunk Overlap

To prevent visible seams or artifacts at the boundaries of video chunks, an overlap strategy should be implemented.

1.  **Implement Overlap Logic**:
    *   Add an `--overlap` command-line argument to specify the number of overlapping frames (e.g., 8 or 12).
    *   The main loop should be adjusted to process chunks with this overlap. For a chunk size `N` and overlap `O`, the stride between chunks will be `N - O`.
    *   For each processed chunk, the output frames corresponding to the overlap region should be discarded (except for the very first chunk). This ensures a smooth transition between chunks.

### Step 5: Recommended Configuration and Usage

After implementing the optimizations, the script should be run with parameters that balance performance and memory usage on a 24GB RTX 4090.

1.  **Optimized Command-Line Usage**:
    *   The following command summarizes the recommended way to run the optimized script:
        ```bash
python tools/restoration_video_streaming.py \
  --config configs/basicvsr_plusplus_reds4.py \
  --checkpoint /path/to/your/checkpoint.pth \
  --input /path/to/input.mov \
  --output /path/to/output.mov \
  --max-seq-len 96 \
  --overlap 8 \
  --writer ffmpeg \
  --device 0
```
2.  **Environment Variable for Memory Allocation**:
    *   For better memory management and to avoid fragmentation, run the script with the following environment variable:
        ```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Step 6: Advanced Optimizations (Future Work)

For even greater performance, the following advanced techniques can be explored:

*   **`torch.compile`**: Wrap the model with `torch.compile(model, mode="max-autotune")` to leverage PyTorch 2.x's JIT compiler. This can provide a 5-15% speedup but requires careful testing.
*   **CUDA Graphs**: For the ultimate performance, the model's forward pass for a fixed-size chunk can be captured into a CUDA Graph to eliminate CPU launch overhead. This is more complex to implement but can yield significant gains.
*   **NVDEC for GPU-side Decoding**: Use `torchaudio.io.StreamReader` to decode video frames directly on the GPU, eliminating the CPU-to-GPU copy entirely.

## Expected Outcome

By implementing steps 1-4, we expect to see a **3-5x improvement in throughput**, from the baseline of ~1.5 fps to **~5.0 fps** on an RTX 4090 for 1024x432 -> 4x upscaling. The GPU utilization should increase from ~60% to over 95%, indicating that the pipeline is no longer I/O-bound but compute-bound, which is the desired state.
