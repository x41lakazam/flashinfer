"""CUTLASS BF16 fused-MoE EXPERT_MAJOR fast path (``dispatch_expert_counts``).

The fast path replaces the sort-based routing prologue with a direct index
computation for callers whose rows are already grouped by expert with a fixed
stride -- the layout NCCL EP's EXPERT_MAJOR dispatch produces. It must be
numerically indistinguishable from the sort path on the real (non-padding) rows.

Padding rows are deliberately *not* compared: the baseline runs the GEMM over
them (producing values from whatever the padding held) while the fast path skips
them entirely. ``combine`` never gathers those rows, so both are correct; only
the real rows carry meaning.
"""

import pytest
import torch

from flashinfer import fused_moe
from flashinfer.utils import get_compute_capability

# Matches _CUTLASS_BF16_ARCHS in flashinfer/fused_moe/api.py.
_SUPPORTED_ARCHS = (89, 90, 100, 103, 107, 110, 120, 121)


def _skip_if_unsupported() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    major, minor = get_compute_capability(torch.device("cuda:0"))
    if major * 10 + minor not in _SUPPORTED_ARCHS:
        pytest.skip(f"CUTLASS BF16 fused MoE unsupported on sm{major}{minor}")


def _make_expert_major_batch(
    num_local_experts, cap, hidden, intermediate, fill_frac, seed=0
):
    """Build an input shaped exactly like EXPERT_MAJOR dispatch output.

    Row ``r`` belongs to expert ``r // cap``; only the first ``counts[e]`` rows of
    each expert's block are real, the rest is padding holding arbitrary values.
    """
    torch.manual_seed(seed)
    device = "cuda"
    num_rows = num_local_experts * cap
    counts = torch.randint(
        1,
        max(2, int(cap * fill_frac)) + 1,
        (num_local_experts,),
        device=device,
        dtype=torch.int32,
    ).clamp(max=cap)

    flat_input = torch.randn(num_rows, hidden, dtype=torch.bfloat16, device=device)
    token_selected_experts = (
        torch.arange(num_local_experts, device=device, dtype=torch.int32)
        .repeat_interleave(cap)
        .reshape(num_rows, 1)
    )
    token_final_scales = torch.ones(num_rows, 1, dtype=torch.float32, device=device)
    w31 = (
        torch.randn(
            num_local_experts,
            2 * intermediate,
            hidden,
            dtype=torch.bfloat16,
            device=device,
        )
        / 5
    )
    w2 = (
        torch.randn(
            num_local_experts, hidden, intermediate, dtype=torch.bfloat16, device=device
        )
        / 5
    )
    return {
        "flat_input": flat_input,
        "token_selected_experts": token_selected_experts,
        "token_final_scales": token_final_scales,
        "w31": w31,
        "w2": w2,
        "counts": counts,
        "cap": cap,
        "num_rows": num_rows,
    }


def _run(batch, *, counts):
    return fused_moe.cutlass_fused_moe(
        batch["flat_input"],
        batch["token_selected_experts"],
        batch["token_final_scales"],
        batch["w31"],
        batch["w2"],
        batch["flat_input"].dtype,
        quant_scales=None,
        dispatch_expert_counts=counts,
    )


def _real_row_mask(batch):
    device = batch["flat_input"].device
    rows = torch.arange(batch["num_rows"], device=device)
    return (rows % batch["cap"]) < batch["counts"][rows // batch["cap"]]


def _as_tensor(out):
    return out[0] if isinstance(out, (list, tuple)) else out


@pytest.mark.parametrize("num_local_experts,cap", [(4, 128), (8, 256), (3, 64)])
@pytest.mark.parametrize("fill_frac", [0.15, 0.9])
def test_fast_path_matches_sort_path(num_local_experts, cap, fill_frac):
    """Real rows must match the sort-based prologue bit-for-bit within tolerance."""
    _skip_if_unsupported()
    batch = _make_expert_major_batch(
        num_local_experts, cap, hidden=512, intermediate=256, fill_frac=fill_frac
    )
    out_base = _as_tensor(_run(batch, counts=None))
    out_fast = _as_tensor(_run(batch, counts=batch["counts"]))

    mask = _real_row_mask(batch)
    assert mask.any(), "test batch must contain at least one real row"
    torch.testing.assert_close(out_base[mask], out_fast[mask], rtol=2e-2, atol=2e-2)


def test_fully_packed_batch_matches_everywhere():
    """With counts == cap there is no padding, so every row must match.

    This pins the boundary case where the fast path and the sort path should agree
    on the whole tensor, not just a subset -- a stronger check than the padded
    cases, which can only compare the real rows.
    """
    _skip_if_unsupported()
    batch = _make_expert_major_batch(
        4, 128, hidden=512, intermediate=256, fill_frac=1.0
    )
    batch["counts"] = torch.full_like(batch["counts"], batch["cap"])

    out_base = _as_tensor(_run(batch, counts=None))
    out_fast = _as_tensor(_run(batch, counts=batch["counts"]))
    torch.testing.assert_close(out_base, out_fast, rtol=2e-2, atol=2e-2)


def test_rejects_top_k_greater_than_one():
    """The fast path's layout contract only holds for pre-routed top_k == 1."""
    _skip_if_unsupported()
    batch = _make_expert_major_batch(
        4, 128, hidden=512, intermediate=256, fill_frac=0.5
    )
    num_rows = batch["num_rows"]
    device = batch["flat_input"].device
    batch["token_selected_experts"] = torch.zeros(
        num_rows, 2, dtype=torch.int32, device=device
    )
    batch["token_final_scales"] = torch.ones(
        num_rows, 2, dtype=torch.float32, device=device
    )

    # TLLM_CHECK_WITH_INFO -> throwRuntimeError -> std::runtime_error, surfaced by
    # TVM-FFI as RuntimeError. Match the message so this cannot pass on an
    # unrelated failure (a shape error, an OOM) and be mistaken for the guard.
    with pytest.raises(RuntimeError, match="experts_per_token == 1"):
        _run(batch, counts=batch["counts"])
