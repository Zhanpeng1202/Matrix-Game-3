"""
Prepare Matrix-Game-3 (action-controlled) inputs for the WBench navigation split.

CPU-only. For each selected WBench case this writes
``<work_dir>/action_paths/case_<id>/{actions.json, prompt.txt}`` and appends an
entry to ``<work_dir>/manifest.json``. The manifest is consumed by
``generate_wbench.py`` (which loads the MG3 model once and renders all cases).

Action conversion reuses WBench's own ``case_to_actions`` helper, which emits the
MG3-native ``{keyboard:[6], mouse:[2]}`` convention (keyboard one-hot
[W,S,A,D,_,_]; mouse [pitch, yaw] in {-0.1, 0, +0.1}). Mapping: one WBench turn ==
one MG3 autoregressive clip, so ``num_iterations = n_turns`` and
``frame_num = 57 + (n_turns - 1) * 40``.

Usage:
    python wbench/prepare_wbench.py \
        --wbench_root /home/builder/workspace/WBench \
        --selection pure_nav
"""
import argparse
import glob
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("prepare_wbench")

FIRST_CLIP_FRAME = 57
LATER_CLIP_NEW_FRAMES = 40


def resolve_image(case: Dict[str, Any], data_dir: str) -> Optional[str]:
    """Find a case's initial image, with a fallback to data/images/case_<id>.jpg."""
    candidates: List[str] = []
    img = case.get("settings", {}).get("initial_image", "")
    if img:
        candidates.append(img if os.path.isabs(img) else os.path.join(data_dir, img))
    candidates.append(os.path.join(data_dir, "images", f"case_{case['id']}.jpg"))
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


def select_cases(cases: List[Dict[str, Any]], selection: str) -> List[Dict[str, Any]]:
    """Filter to the desired navigation subset."""
    def is_nav(i):
        return i.get("type") == "navigation"

    out = []
    for c in cases:
        inters = c.get("interactions", [])
        if not inters:
            continue
        if selection == "pure_nav":
            if all(is_nav(i) for i in inters):
                out.append(c)
        elif selection == "any_nav":
            if any(is_nav(i) for i in inters):
                out.append(c)
        else:
            raise ValueError(f"unknown selection: {selection}")
    return out


def build_prompt(case: Dict[str, Any]) -> str:
    """Single static per-case prompt: environment_prompt + character_prompt.

    MG3 takes one prompt for the whole clip and lets the keyboard/mouse signal
    drive motion, so we use the scene + subject description (AGENT.md section 5).
    """
    parts = [
        str(case.get("environment_prompt", "")).strip(),
        str(case.get("character_prompt", "")).strip(),
    ]
    return " ".join(p for p in parts if p)


def main():
    ap = argparse.ArgumentParser(description="Prepare MG3 action-controlled inputs for WBench")
    ap.add_argument("--wbench_root", default="/home/builder/workspace/WBench",
                    help="WBench repo root (provides data/ and src/).")
    ap.add_argument("--data_dir", default=None, help="Defaults to <wbench_root>/data")
    ap.add_argument("--work_dir", default=None,
                    help="Defaults to <wbench_root>/work_dirs/matrix_game_3")
    ap.add_argument("--selection", choices=["pure_nav", "any_nav"], default="pure_nav")
    ap.add_argument("--duration", type=float, default=4.0,
                    help="Nominal per-turn duration passed to WBench case_to_actions.")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N cases (debug).")
    ap.add_argument("--case_ids", default=None,
                    help="Comma-separated case ids to restrict to (e.g. '1,2,3').")
    args = ap.parse_args()

    wbench_root = os.path.abspath(args.wbench_root)
    data_dir = args.data_dir or os.path.join(wbench_root, "data")
    work_dir = args.work_dir or os.path.join(wbench_root, "work_dirs", "matrix_game_3")

    # Import WBench's MG3-style action converter from the repo root.
    if wbench_root not in sys.path:
        sys.path.insert(0, wbench_root)
    try:
        from src.models.action.actions import case_to_actions
    except Exception as e:  # noqa: BLE001
        logger.error(f"Could not import WBench case_to_actions from {wbench_root}: {e}")
        raise

    cases_dir = os.path.join(data_dir, "cases")
    files = sorted(glob.glob(os.path.join(cases_dir, "case_*.json")),
                   key=lambda p: int(os.path.basename(p)[5:-5]))
    if not files:
        raise FileNotFoundError(f"No case_*.json under {cases_dir}")
    cases = [json.load(open(f)) for f in files]

    selected = select_cases(cases, args.selection)
    if args.case_ids:
        wanted = {s.strip() for s in args.case_ids.split(",") if s.strip()}
        selected = [c for c in selected if str(c["id"]) in wanted]
    if args.limit:
        selected = selected[: args.limit]
    logger.info(f"Loaded {len(cases)} cases; selected {len(selected)} ({args.selection})")

    out_dir = os.path.join(work_dir, "action_paths")
    os.makedirs(out_dir, exist_ok=True)

    manifest: List[Dict[str, Any]] = []
    skipped = []
    for c in selected:
        cid = c["id"]
        image = resolve_image(c, data_dir)
        if image is None:
            logger.warning(f"case_{cid}: image not found, skipping")
            skipped.append(cid)
            continue

        conv = case_to_actions(c, duration=args.duration)
        turns = conv["actions"]  # per-turn {turn, tokens, keyboard, mouse, duration}
        n_turns = len(turns)
        if n_turns == 0:
            logger.warning(f"case_{cid}: no actions, skipping")
            skipped.append(cid)
            continue

        num_iterations = n_turns
        frame_num = FIRST_CLIP_FRAME + (num_iterations - 1) * LATER_CLIP_NEW_FRAMES
        prompt = build_prompt(c)

        case_dir = os.path.join(out_dir, f"case_{cid}")
        os.makedirs(case_dir, exist_ok=True)
        with open(os.path.join(case_dir, "actions.json"), "w") as fp:
            json.dump(turns, fp, indent=2)
        with open(os.path.join(case_dir, "prompt.txt"), "w") as fp:
            fp.write(prompt)

        manifest.append({
            "id": cid,
            "image": image,
            "action_path": os.path.abspath(case_dir),
            "prompt": prompt,
            "n_turns": n_turns,
            "num_iterations": num_iterations,
            "frame_num": frame_num,
            "perspective": c.get("settings", {}).get("perspective", ""),
            "nav_cate": c.get("nav_cate", ""),
            "actions": turns,
        })

    os.makedirs(work_dir, exist_ok=True)
    manifest_path = os.path.join(work_dir, "manifest.json")
    with open(manifest_path, "w") as fp:
        json.dump(manifest, fp, indent=2)

    if manifest:
        fn = [m["frame_num"] for m in manifest]
        logger.info(f"Wrote {len(manifest)} cases -> {out_dir}")
        logger.info(f"frame_num: min={min(fn)} max={max(fn)} (= 57 + (n_turns-1)*40)")
    if skipped:
        logger.info(f"skipped: {skipped}")
    logger.info(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
