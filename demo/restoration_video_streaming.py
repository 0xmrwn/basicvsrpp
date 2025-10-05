"""
Streaming BasicVSR++ video super-resolution

Reads an input video in chunks, runs BasicVSR++ inference, and writes output
frames immediately to disk as an image sequence to avoid OOM on long videos.

Test command:
  source /opt/bsvrpp-venv/bin/activate && python /workspace/restoration_video_streaming.py \
    --config /opt/BasicVSR_PlusPlus/configs/basicvsr_plusplus_reds4.py \
    --checkpoint /workspace/models/basicvsr_plusplus_reds4.pth \
    --input /workspace/tmp_pre/main_SAMPLE_1024x432_25p_bt709.mov \
    --output /workspace/tmp_sr \
    --max-seq-len 250 \
    --device 0
"""

import os
import sys
import argparse
import warnings
from typing import List, Tuple, Optional

import torch
import mmcv

try:
    from mmcv import Config
except Exception:
    from mmengine.config import Config  # fallback for newer stacks

from mmedit.apis import init_model
from mmedit.core import tensor2img

# Try to import Compose from mmedit (preferred), else fall back to mmcv/mmengine variants.
try:
    from mmedit.datasets.pipelines import Compose
except Exception:
    try:
        from mmcv.transforms import Compose  # mmcv>=2
    except Exception:
        try:
            from mmengine.dataset import Compose  # mmengine fallback
        except Exception:
            Compose = None  # Will fallback to manual preprocessing


def build_in_memory_pipeline(cfg: Config) -> Optional[object]:
    """Build a minimal in-memory preprocessing pipeline based on demo_pipeline.

    We keep only steps that operate on in-memory frames ('lq') without requiring disk I/O:
      - RescaleToZeroOne
      - FramesToTensor
    """
    pipeline_cfg = None
    if hasattr(cfg, 'demo_pipeline'):
        pipeline_cfg = cfg.demo_pipeline
    elif hasattr(cfg, 'test_pipeline'):
        pipeline_cfg = cfg.test_pipeline

    if pipeline_cfg is None or Compose is None:
        return None

    keep_types = {'RescaleToZeroOne', 'FramesToTensor'}
    minimal = []
    for step in pipeline_cfg:
        stype = step.get('type', '')
        if stype in keep_types:
            step = step.copy()
            # Ensure these steps operate on key 'lq' (our in-memory frame list)
            if 'keys' in step:
                step['keys'] = ['lq']
            minimal.append(step)

    if not minimal:
        return None
    return Compose(minimal)


def frames_to_lq_tensor(frames: List, preprocess=None) -> torch.Tensor:
    """Convert a list of HxWx3 uint8 BGR frames into a tensor (1, T, 3, H, W) in [0,1]."""
    if preprocess is not None:
        results = dict(lq=frames)
        results = preprocess(results)
        lq = results.get('lq', None)
        if lq is None:
            raise RuntimeError('Preprocess pipeline did not produce "lq".')
        if isinstance(lq, list) and isinstance(lq[0], torch.Tensor):
            # list[T] of (3,H,W)
            lq = torch.stack(lq, dim=0)  # (T, 3, H, W)
        elif isinstance(lq, torch.Tensor) and lq.dim() == 4:
            # already (T, 3, H, W)
            pass
        else:
            raise RuntimeError('Unexpected "lq" type from pipeline.')
        lq = lq.unsqueeze(0)  # (1, T, 3, H, W)
        return lq

    # Manual fallback if no pipeline is available (RescaleToZeroOne + FramesToTensor)
    # Convert to torch, scale to [0,1], reorder to (T, 3, H, W), then add batch dim.
    tensors = []
    for img in frames:
        # img: HxWxC (BGR, uint8)
        t = torch.from_numpy(img).float() / 255.0  # HWC
        t = t.permute(2, 0, 1).contiguous()  # CHW
        tensors.append(t)
    lq = torch.stack(tensors, dim=0)  # (T, 3, H, W)
    lq = lq.unsqueeze(0)  # (1, T, 3, H, W)
    return lq


def ensure_dir(path: str):
    mmcv.mkdir_or_exist(path)


def parse_args():
    parser = argparse.ArgumentParser(description='Streaming BasicVSR++ video super-resolution')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--input', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, required=True, help='Output directory for PNG frames')
    parser.add_argument('--max-seq-len', type=int, default=250, help='Chunk size (frames) to process at once')
    parser.add_argument('--device', type=int, default=0, help='CUDA device index (use -1 for CPU)')
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.config):
        raise FileNotFoundError(f'Config not found: {args.config}')
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f'Input video not found: {args.input}')

    ensure_dir(args.output)

    # Load config
    cfg = Config.fromfile(args.config)

    # Optional: enforce recurrent mode by setting window_size < 0 if present in test_cfg
    try:
        if hasattr(cfg, 'model') and hasattr(cfg.model, 'test_cfg'):
            test_cfg = cfg.model.test_cfg
            if isinstance(test_cfg, dict) and 'window_size' in test_cfg and test_cfg['window_size'] is not None:
                if test_cfg['window_size'] >= 0:
                    test_cfg['window_size'] = -1
    except Exception:
        warnings.warn('Could not update test_cfg.window_size to recurrent mode; proceeding with defaults.')

    # Init model
    device_str = f'cuda:{args.device}' if args.device >= 0 and torch.cuda.is_available() else 'cpu'
    model = init_model(cfg, args.checkpoint, device=device_str)
    model.eval()

    # Build in-memory preprocessing pipeline
    preprocess = build_in_memory_pipeline(cfg)

    # Prepare video reader
    vr = mmcv.VideoReader(args.input)
    total_frames = len(vr)
    if total_frames == 0:
        raise RuntimeError(f'No frames read from video: {args.input}')

    print(f'Input: {args.input}')
    print(f'Total frames: {total_frames}')
    print(f'Output dir: {args.output}')
    print(f'Device: {device_str}')
    print(f'Chunk size: {args.max_seq_len}')

    # Optional performance tweak
    torch.set_grad_enabled(False)
    if device_str.startswith('cuda'):
        torch.backends.cudnn.benchmark = True

    # Process in chunks
    frame_counter = 0
    for chunk_start in range(0, total_frames, args.max_seq_len):
        chunk_end = min(chunk_start + args.max_seq_len, total_frames)
        # Read frames [chunk_start, chunk_end)
        frames = [vr[i] for i in range(chunk_start, chunk_end)]

        # Preprocess -> (1, T, 3, H, W), float32 in [0,1]
        lq = frames_to_lq_tensor(frames, preprocess=preprocess)  # CPU tensor
        T = lq.shape[1]

        # Inference
        with torch.inference_mode() if hasattr(torch, 'inference_mode') else torch.no_grad():
            out = model(lq=lq.to(device_str, non_blocking=True), test_mode=True)
            # Extract predictions
            if isinstance(out, dict):
                out = out.get('output', out.get('pred', out))
            # out could be list[T](3,H_up,W_up) or tensor [1,T,3,H_up,W_up] or [T,3,H_up,W_up]
            if isinstance(out, (list, tuple)):
                pred_list = [o.detach().cpu() for o in out]
            elif isinstance(out, torch.Tensor):
                out = out.detach().cpu()
                if out.dim() == 5 and out.size(0) == 1:
                    out = out.squeeze(0)  # (T,3,H,W)
                if out.dim() == 4:
                    pred_list = [out[i] for i in range(out.size(0))]
                else:
                    raise RuntimeError(f'Unexpected output tensor shape: {tuple(out.shape)}')
            else:
                raise RuntimeError(f'Unexpected output type: {type(out)}')

        # Write frames immediately
        for i, pred in enumerate(pred_list):
            img = tensor2img(pred)  # uint8 BGR, HxWx3
            out_idx = chunk_start + i
            out_path = os.path.join(args.output, f'{out_idx:08d}.png')
            mmcv.imwrite(img, out_path)

        print(f'Processed chunk {chunk_start}-{chunk_end - 1}, wrote {len(pred_list)} frames.')

        # Free memory
        del lq, out, pred_list, frames
        if device_str.startswith('cuda'):
            torch.cuda.empty_cache()

        frame_counter += (chunk_end - chunk_start)

    print(f'Done. Wrote {frame_counter} frames to {args.output}')


if __name__ == '__main__':
    sys.exit(main())