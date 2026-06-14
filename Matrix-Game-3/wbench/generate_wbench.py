"""
Generate Matrix-Game-3 (action-controlled) videos for the WBench navigation split.

Loads the MG3 model ONCE and renders every case in the manifest produced by
``prepare_wbench.py`` into ``<save_dir>/case_<id>_combined.mp4`` (the layout
WBench evaluation expects under ``work_dirs/<model>/videos/``).

Each case is one continuous causal rollout: one WBench turn == one MG3
autoregressive clip (num_iterations = n_turns), with the model's long-horizon
memory spanning the whole case. Generation only -- no scoring.

Usage (8x GPU, sequence-parallel):
    torchrun --nproc_per_node=8 wbench/generate_wbench.py \
        --ckpt_dir Matrix-Game-3.0 \
        --manifest /home/builder/workspace/WBench/work_dirs/matrix_game_3/manifest.json \
        --save_dir /home/builder/workspace/WBench/work_dirs/matrix_game_3/videos \
        --size 704*1280 --ulysses_size 8 --dit_fsdp --t5_fsdp \
        --use_int8 --num_inference_steps 3 \
        --vae_type mg_lightvae --lightvae_pruning_rate 0.5 --compile_vae \
        --fa_version 3 --resume
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # MG3 repo root

import torch
import torch.distributed as dist
from PIL import Image

from wan.configs import WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from utils.misc import set_seed
from pipeline.inference_wbench_pipeline import MatrixGame3WBenchPipeline


def parse_args():
    ap = argparse.ArgumentParser(description="Matrix-Game-3 x WBench batch generation (action-controlled)")
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--size", default="704*1280")

    # Distributed / parallel
    ap.add_argument("--ulysses_size", type=int, default=None,
                    help="Sequence-parallel size for DiT. Defaults to WORLD_SIZE.")
    ap.add_argument("--dit_fsdp", action="store_true", default=False)
    ap.add_argument("--t5_fsdp", action="store_true", default=False)
    ap.add_argument("--t5_cpu", action="store_true", default=False)
    ap.add_argument("--convert_model_dtype", action="store_true", default=False)

    # Sampling
    ap.add_argument("--num_inference_steps", type=int, default=3)
    ap.add_argument("--sample_shift", type=float, default=None)
    ap.add_argument("--sample_guide_scale", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_base_model", action="store_true", default=False)

    # VAE / quantization / attention
    ap.add_argument("--vae_type", default="mg_lightvae", choices=["wan", "mg_lightvae", "mg_lightvae_v2"])
    ap.add_argument("--lightvae_pruning_rate", type=float, default=0.5)
    ap.add_argument("--compile_vae", action="store_true", default=False)
    ap.add_argument("--use_int8", action="store_true", default=False)
    ap.add_argument("--fa_version", type=str, default="3", choices=["0", "2", "3"])

    # Output / batching
    ap.add_argument("--fps", type=int, default=16, help="Playback fps for saved mp4.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true", default=False,
                    help="Skip cases whose output mp4 already exists.")
    return ap.parse_args()


def init_distributed(ulysses_size):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://",
                                rank=rank, world_size=world_size)
    if ulysses_size > 1:
        assert ulysses_size == world_size, \
            f"ulysses_size ({ulysses_size}) must equal WORLD_SIZE ({world_size})"
        init_distributed_group()
    level = logging.INFO if rank == 0 else logging.ERROR
    logging.basicConfig(level=level, format="[%(asctime)s] %(message)s",
                        handlers=[logging.StreamHandler(stream=sys.stdout)])
    return rank, local_rank, world_size


def main():
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if args.ulysses_size is None:
        args.ulysses_size = world_size

    # Mirror generate.py: single-rank compute (ulysses_size<=1) cannot use FSDP.
    if args.ulysses_size <= 1 and (args.t5_fsdp or args.dit_fsdp):
        args.t5_fsdp = False
        args.dit_fsdp = False

    rank, local_rank, world_size = init_distributed(args.ulysses_size)
    cfg = WAN_CONFIGS["matrix_game3"]
    shift = args.sample_shift if args.sample_shift is not None else cfg.sample_shift

    with open(args.manifest) as fp:
        manifest = json.load(fp)
    if args.limit:
        manifest = manifest[: args.limit]
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        logging.info(f"Manifest: {len(manifest)} cases | size={args.size} "
                     f"shift={shift} steps={args.num_inference_steps} ulysses={args.ulysses_size}")

    # Build a generate.py-compatible args object for the pipeline constructor / VAE config.
    pipe_args = SimpleNamespace(
        size=args.size,
        ckpt_dir=args.ckpt_dir,
        output_dir=args.save_dir,
        use_int8=args.use_int8,
        verify_quant=False,
        use_async_vae=False,
        num_iterations=1,            # per-case value is passed to generate_case
        compile_vae=args.compile_vae,
        async_vae_warmup_iters=0,
        vae_type=args.vae_type,
        lightvae_pruning_rate=args.lightvae_pruning_rate,
        save_name="wbench",
        fa_version=args.fa_version,
    )

    # Cold start: build the pipeline once.
    pipe = MatrixGame3WBenchPipeline(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
        args=pipe_args,
        fa_version=args.fa_version,
        use_base_model=args.use_base_model,
    )
    if rank == 0:
        pipe._log_flash_attention_config(pipe_args)

    n_ok, n_skip, n_fail = 0, 0, 0
    t_start = time.perf_counter()
    for idx, m in enumerate(manifest):
        cid = m["id"]
        out_path = os.path.join(args.save_dir, f"case_{cid}_combined.mp4")
        if args.resume and os.path.exists(out_path):
            n_skip += 1
            if rank == 0:
                logging.info(f"[{idx+1}/{len(manifest)}] case_{cid}: SKIP (exists)")
            continue

        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
        t0 = time.perf_counter()

        set_seed(args.seed)
        img = Image.open(m["image"]).convert("RGB")
        try:
            pipe.generate_case(
                text=m["prompt"],
                pil_image=img,
                actions=m["actions"],
                save_path=out_path,
                size=args.size,
                num_inference_steps=args.num_inference_steps,
                shift=shift,
                guide_scale=args.sample_guide_scale,
                seed=args.seed,
                use_base_model=args.use_base_model,
                fps=args.fps,
                args=pipe_args,
            )
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            if rank == 0:
                logging.error(f"[{idx+1}/{len(manifest)}] case_{cid}: FAIL {e}")
            # Keep ranks aligned before moving on.
            if dist.is_initialized():
                dist.barrier()
            continue

        dt = time.perf_counter() - t0
        if rank == 0:
            n_ok += 1
            logging.info(f"[{idx+1}/{len(manifest)}] case_{cid}: OK "
                         f"({m['n_turns']} turns, {m['frame_num']} frames, {dt:.1f}s) -> {out_path}")

    if rank == 0:
        total = time.perf_counter() - t_start
        logging.info(f"Done in {total/60:.1f} min - ok={n_ok} skip={n_skip} fail={n_fail}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
