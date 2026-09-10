"""Correctness + microbenchmark for the EXPERT_MAJOR dispatch fast path.

Compares the default CUTLASS fused_moe sort-based prologue
(threeStepBuildExpertMapsSortFirstToken / fusedBuildExpertMapsSortFirstToken +
expandInputRowsKernel over the full padded [num_local_experts*cap, hidden] buffer)
against the new ``dispatch_expert_counts`` fast path (see
csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh,
buildExpertMapsFromKnownCounts) that skips the sort and skips gathering /
computing padding rows, for a synthetic batch shaped exactly like NCCL EP's
EXPERT_MAJOR dispatch output (flashinfer/moe_ep/backends/split/kernel/fused_moe/bridge.py).

Usage:
    python benchmarks/bench_ep_layout_sort_redundancy.py
    python benchmarks/bench_ep_layout_sort_redundancy.py --stage A --csv stage_a.csv --gpu-tag h100_gaia
    python benchmarks/bench_ep_layout_sort_redundancy.py --stage B --skewed --csv stage_b_skewed.csv --gpu-tag gb200_rachel

See NCCL_EP_LAYOUT_REDUNDANCY_SWEEP_PLAN.md for the staged sweep design and value choices.
"""

from __future__ import annotations

import argparse
import csv as csv_module
import itertools
import statistics
from dataclasses import dataclass
from typing import Optional

import torch

import flashinfer.fused_moe as fused_moe
from flashinfer.testing import bench_gpu_time

# Canonical LL BF16 dispatch widths (flashinfer/moe_ep/backends/split/kernel/fused_moe/bridge.py:38)
BF16_WIDTHS = (2048, 2560, 4096, 5120, 6144, 7168, 8192)


@dataclass
class Shape:
    num_local_experts: int
    hidden: int
    intermediate: int
    fill_frac: float  # target real tokens / cap, per expert
    # cap is either given directly (matches the original 5 hand-picked shapes) or derived
    # as max_tokens_per_rank * world_size (EpLayout.EXPERT_MAJOR's actual construction,
    # flashinfer/moe_ep/config.py). Set exactly one of {cap, (world_size & max_tokens_per_rank)}.
    cap: Optional[int] = None
    world_size: Optional[int] = None
    max_tokens_per_rank: Optional[int] = None
    skewed: bool = (
        False  # one-hot-heavy per-expert count distribution instead of uniform
    )

    def __post_init__(self):
        if self.cap is None:
            if self.world_size is None or self.max_tokens_per_rank is None:
                raise ValueError(
                    "Shape needs either cap= or both world_size= and max_tokens_per_rank="
                )
            self.cap = self.world_size * self.max_tokens_per_rank


def make_batch(shape: Shape, device: str = "cuda", seed: int = 0):
    torch.manual_seed(seed)
    E, cap, H, I = shape.num_local_experts, shape.cap, shape.hidden, shape.intermediate

    if not shape.skewed:
        counts = torch.randint(
            max(1, int(cap * shape.fill_frac * 0.5)),
            max(2, int(cap * shape.fill_frac)) + 1,
            (E,),
            device=device,
            dtype=torch.int32,
        ).clamp(max=cap)
    else:
        # Skewed load: same target mean (fill_frac * cap) as the uniform case, but
        # concentrated on a minority of "hot" experts (near-full) while the rest sit
        # near-empty -- a more realistic stand-in for imbalanced MoE routing than a
        # uniform random range around the mean.
        target_mean = max(1.0, cap * shape.fill_frac)
        num_hot = max(1, E // 4)
        weights = torch.ones(E, device=device, dtype=torch.float64)
        weights[:num_hot] = 8.0  # hot experts get ~8x the share of a cold expert
        weights = weights[torch.randperm(E, device=device)]
        counts = (weights / weights.sum() * target_mean * E).round().to(torch.int32)
        counts = counts.clamp(min=1, max=cap)

    # EXPERT_MAJOR dispatch output: [num_local_experts, cap, hidden], front-packed real
    # rows per expert, garbage padding -- exactly bridge.py's build_activation_pack input.
    expert_tensors = torch.randn(E, cap, H, dtype=torch.bfloat16, device=device)

    num_rows = E * cap
    token_selected_experts = (
        torch.arange(E, device=device, dtype=torch.int32)
        .repeat_interleave(cap)
        .reshape(num_rows, 1)
    )
    token_final_scales = torch.ones(num_rows, 1, dtype=torch.float32, device=device)

    w31_weight = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device=device) / 5
    w2_weight = torch.randn(E, H, I, dtype=torch.bfloat16, device=device) / 5

    flat_input = expert_tensors.reshape(num_rows, H)
    return {
        "flat_input": flat_input,
        "token_selected_experts": token_selected_experts,
        "token_final_scales": token_final_scales,
        "w31_weight": w31_weight,
        "w2_weight": w2_weight,
        "counts": counts,
        "cap": cap,
        "num_rows": num_rows,
    }


def run_baseline(batch):
    return fused_moe.cutlass_fused_moe(
        batch["flat_input"],
        batch["token_selected_experts"],
        batch["token_final_scales"],
        batch["w31_weight"],
        batch["w2_weight"],
        batch["flat_input"].dtype,
        quant_scales=None,
        dispatch_expert_counts=None,
    )


def run_fast_path(batch):
    return fused_moe.cutlass_fused_moe(
        batch["flat_input"],
        batch["token_selected_experts"],
        batch["token_final_scales"],
        batch["w31_weight"],
        batch["w2_weight"],
        batch["flat_input"].dtype,
        quant_scales=None,
        dispatch_expert_counts=batch["counts"],
    )


def real_row_mask(batch) -> torch.Tensor:
    cap = batch["cap"]
    counts = batch["counts"]
    row_expert = (
        torch.arange(batch["num_rows"], device=batch["flat_input"].device) // cap
    )
    local = torch.arange(batch["num_rows"], device=batch["flat_input"].device) % cap
    return local < counts[row_expert]


def estimated_bytes(shape: Shape) -> int:
    """Rough worst-case device memory for one shape's tensors (activations + weights,
    baseline and fast-path outputs), used to skip combinations that can't fit before
    hitting a hard CUDA OOM mid-sweep."""
    E, cap, H, I = shape.num_local_experts, shape.cap, shape.hidden, shape.intermediate
    num_rows = E * cap
    activation = (
        num_rows * H * 2
    )  # bf16 input, ignore intermediate/duplicated output copies
    weights = E * (2 * I * H + H * I) * 2  # w31 + w2, bf16
    return int(
        (activation + weights) * 3
    )  # x3 fudge factor for intermediates/duplicated buffers


def filter_shapes_for_memory(
    shapes: list[Shape], budget_bytes: int
) -> tuple[list[Shape], list[Shape]]:
    keep, skipped = [], []
    for s in shapes:
        (keep if estimated_bytes(s) <= budget_bytes else skipped).append(s)
    return keep, skipped


def check_correctness(shapes: list[Shape]) -> bool:
    ok = True
    for shape in shapes:
        try:
            batch = make_batch(shape)
            out_base = run_baseline(batch)
            out_fast = run_fast_path(batch)
        except (torch.cuda.OutOfMemoryError, MemoryError):
            torch.cuda.empty_cache()
            print(
                f"[correctness] E={shape.num_local_experts} cap={shape.cap} H={shape.hidden}: "
                f"SKIP (OOM)"
            )
            continue
        mask = real_row_mask(batch)
        base_real = (
            out_base[0][mask] if isinstance(out_base, (list, tuple)) else out_base[mask]
        )
        fast_real = (
            out_fast[0][mask] if isinstance(out_fast, (list, tuple)) else out_fast[mask]
        )
        try:
            torch.testing.assert_close(base_real, fast_real, rtol=2e-2, atol=2e-2)
            status = "PASS"
        except AssertionError as e:
            status = f"FAIL ({e})"
            ok = False
        print(
            f"[correctness] E={shape.num_local_experts} cap={shape.cap} H={shape.hidden} "
            f"real_rows={int(mask.sum())}/{batch['num_rows']}: {status}"
        )
        del batch, out_base, out_fast
        torch.cuda.empty_cache()
    return ok


def bench(
    shapes: list[Shape],
    repeat_iters: int = 30,
    dry_run_iters: int = 10,
    gpu_tag: str = "",
):
    rows = []
    for shape in shapes:
        try:
            batch = make_batch(shape)

            t_base = bench_gpu_time(
                lambda b=batch: run_baseline(b),
                dry_run_iters=dry_run_iters,
                repeat_iters=repeat_iters,
            )
            t_fast = bench_gpu_time(
                lambda b=batch: run_fast_path(b),
                dry_run_iters=dry_run_iters,
                repeat_iters=repeat_iters,
            )
        except (torch.cuda.OutOfMemoryError, MemoryError):
            torch.cuda.empty_cache()
            print(
                f"E={shape.num_local_experts:3d} cap={shape.cap:4d} H={shape.hidden:5d}: SKIP (OOM)"
            )
            rows.append(
                {
                    "gpu_tag": gpu_tag,
                    "num_local_experts": shape.num_local_experts,
                    "cap": shape.cap,
                    "world_size": shape.world_size,
                    "max_tokens_per_rank": shape.max_tokens_per_rank,
                    "hidden": shape.hidden,
                    "intermediate": shape.intermediate,
                    "fill_frac": shape.fill_frac,
                    "skewed": shape.skewed,
                    "num_rows": shape.num_local_experts * shape.cap,
                    "real_rows": None,
                    "baseline_ms": None,
                    "fast_path_ms": None,
                    "speedup": None,
                    "note": "OOM",
                }
            )
            continue

        med_base = statistics.median(t_base)
        med_fast = statistics.median(t_fast)
        speedup = med_base / med_fast if med_fast > 0 else float("nan")
        real_rows = int(real_row_mask(batch).sum())
        row = {
            "gpu_tag": gpu_tag,
            "num_local_experts": shape.num_local_experts,
            "cap": shape.cap,
            "world_size": shape.world_size,
            "max_tokens_per_rank": shape.max_tokens_per_rank,
            "hidden": shape.hidden,
            "intermediate": shape.intermediate,
            "fill_frac": shape.fill_frac,
            "skewed": shape.skewed,
            "num_rows": batch["num_rows"],
            "real_rows": real_rows,
            "baseline_ms": med_base,
            "fast_path_ms": med_fast,
            "speedup": speedup,
            "note": "",
        }
        rows.append(row)
        del batch
        torch.cuda.empty_cache()
        print(
            f"E={shape.num_local_experts:3d} cap={shape.cap:4d} (ws={shape.world_size} "
            f"mtpr={shape.max_tokens_per_rank}) H={shape.hidden:5d} skewed={shape.skewed} "
            f"real={real_rows:5d}/{row['num_rows']:5d}  "
            f"baseline={med_base:8.4f} ms  fast_path={med_fast:8.4f} ms  "
            f"speedup={speedup:5.2f}x"
        )
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {path}")


# ---------------------------------------------------------------------------
# Staged sweep grids, per NCCL_EP_LAYOUT_REDUNDANCY_SWEEP_PLAN.md Section 3.
# ---------------------------------------------------------------------------


def default_shapes() -> list[Shape]:
    """The original 5 hand-picked shapes used in the first exec-report benchmark."""
    return [
        Shape(
            num_local_experts=8, cap=256, hidden=2048, intermediate=4096, fill_frac=0.5
        ),
        Shape(
            num_local_experts=8, cap=256, hidden=4096, intermediate=8192, fill_frac=0.5
        ),
        Shape(
            num_local_experts=8, cap=256, hidden=7168, intermediate=8192, fill_frac=0.5
        ),
        Shape(
            num_local_experts=32,
            cap=128,
            hidden=4096,
            intermediate=8192,
            fill_frac=0.25,
        ),
        Shape(
            num_local_experts=8, cap=512, hidden=4096, intermediate=8192, fill_frac=0.1
        ),
    ]


def stage_a_shapes(skewed: bool = False) -> list[Shape]:
    """Coarse full factorial: 3 levels/axis, 81 shapes. Finds the region of interest."""
    experts = (8, 32, 64)
    world_sizes = (8, 32, 128)
    hiddens = (2048, 4096, 7168)
    fill_fracs = (0.15, 0.5, 0.9)
    max_tokens_per_rank = 128
    shapes = []
    for E, ws, H, ff in itertools.product(experts, world_sizes, hiddens, fill_fracs):
        shapes.append(
            Shape(
                num_local_experts=E,
                world_size=ws,
                max_tokens_per_rank=max_tokens_per_rank,
                hidden=H,
                intermediate=2 * H,
                fill_frac=ff,
                skewed=skewed,
            )
        )
    return shapes


def stage_b_shapes(skewed: bool = False) -> list[Shape]:
    """Targeted refinement around the known (E=32, cap=128, hidden=4096) regression.

    NOTE: Stage A's grid could not reach cap=128 (its smallest combo is
    world_size=8 * max_tokens_per_rank=128 = 1024), so it neither confirmed nor
    refuted the original regression. max_tokens_per_rank here is deliberately
    smaller (16/32/64/128, not 128/256) and world_size includes 4, so this grid
    actually covers cap=128 (world_size=4,mtpr=32 or world_size=8,mtpr=16) and
    cap=256/512 (the other two originally hand-picked shapes), not just values
    an order of magnitude larger.
    """
    # fill_frac kept coarse here (Stage A already nailed down that speedup scales with
    # it); the point of Stage B is covering the cap range Stage A couldn't reach, not
    # re-establishing the fill_frac relationship at finer resolution.
    experts = (16, 24, 32, 40, 48, 56, 64)
    world_sizes = (4, 8, 16)
    fill_fracs = (0.1, 0.5, 0.9)
    max_tokens_per_ranks = (16, 32, 64, 128)
    hidden = 4096
    shapes = []
    for E, ws, ff, mtpr in itertools.product(
        experts, world_sizes, fill_fracs, max_tokens_per_ranks
    ):
        shapes.append(
            Shape(
                num_local_experts=E,
                world_size=ws,
                max_tokens_per_rank=mtpr,
                hidden=hidden,
                intermediate=2 * hidden,
                fill_frac=ff,
                skewed=skewed,
            )
        )
    return shapes


def stage_c_shapes() -> list[Shape]:
    """Realism check: skewed distribution + remaining hidden sizes at the Stage-B worst point."""
    remaining_hiddens = (2560, 5120, 6144, 8192)
    shapes = []
    for H in remaining_hiddens:
        for skewed in (False, True):
            shapes.append(
                Shape(
                    num_local_experts=32,
                    world_size=8,
                    max_tokens_per_rank=128,
                    hidden=H,
                    intermediate=2 * H,
                    fill_frac=0.25,
                    skewed=skewed,
                )
            )
    return shapes


STAGE_BUILDERS = {
    "A": stage_a_shapes,
    "B": stage_b_shapes,
    "C": lambda skewed=False: stage_c_shapes(),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--repeat-iters", type=int, default=30)
    parser.add_argument(
        "--stage",
        choices=["default", "A", "B", "C"],
        default="default",
        help="Shape grid to run; see NCCL_EP_LAYOUT_REDUNDANCY_SWEEP_PLAN.md.",
    )
    parser.add_argument(
        "--skewed", action="store_true", help="Use skewed per-expert load distribution."
    )
    parser.add_argument(
        "--csv", type=str, default=None, help="Path to write per-shape results as CSV."
    )
    parser.add_argument(
        "--gpu-tag",
        type=str,
        default="",
        help="Label for which GPU/cluster produced this run.",
    )
    parser.add_argument(
        "--mem-budget-gb",
        type=float,
        default=None,
        help="Skip shapes whose rough estimated tensor footprint exceeds this many GB "
        "(defaults to 60%% of the detected GPU's total memory).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")
    gpu_name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(
        f"GPU: {gpu_name}  tag={args.gpu_tag}  stage={args.stage}  skewed={args.skewed}"
    )

    if args.stage == "default":
        shapes = default_shapes()
    else:
        shapes = STAGE_BUILDERS[args.stage](skewed=args.skewed)

    mem_budget_gb = (
        args.mem_budget_gb if args.mem_budget_gb is not None else 0.6 * total_mem_gb
    )
    shapes, skipped = filter_shapes_for_memory(shapes, int(mem_budget_gb * 1e9))
    print(
        f"Running {len(shapes)} shape(s) (skipped {len(skipped)} over the "
        f"{mem_budget_gb:.1f} GB estimated-memory budget on this {total_mem_gb:.0f} GB GPU)."
    )
    for s in skipped:
        print(
            f"  [skip: too big] E={s.num_local_experts} cap={s.cap} H={s.hidden} "
            f"(~{estimated_bytes(s) / 1e9:.1f} GB estimated)"
        )

    if not args.skip_correctness:
        ok = check_correctness(shapes)
        if not ok:
            raise SystemExit("Correctness check FAILED -- not proceeding to benchmark.")
        print()

    rows = bench(
        shapes, repeat_iters=args.repeat_iters, gpu_tag=args.gpu_tag or gpu_name
    )

    if args.csv:
        write_csv(rows, args.csv)
