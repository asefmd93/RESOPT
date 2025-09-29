import os
import glob
import pickle
import argparse
from typing import Dict, List, Tuple
import numpy as np

EPS = 1e-12  # tiny positive floor to avoid zeros/NaNs anywhere

# Per-layer fields from LLM_profiler we aggregate
FIELDS = ["ops_fwd", "ops_bwd", "bytes_mv_fwd", "bytes_mv_bwd", "param_bytes"]


def _is_finite(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def _to_pos(x) -> float:
    """Convert to strictly positive finite float (NaN/Inf/<=0 -> EPS)."""
    if not _is_finite(x):
        return EPS
    x = float(x)
    return x if x > 0.0 else EPS


def load_layers(path: str) -> List[Dict]:
    """Load and sanitize the per-layer list from an LLM_profiler PKL."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "layers" in data:
        layers = data["layers"]
    elif isinstance(data, list):
        layers = data
    else:
        raise ValueError(f"{path}: unsupported PKL structure; expected dict with 'layers' or a list")

    clean_layers = []
    for lyr in layers:
        if not isinstance(lyr, dict):
            continue
        entry = {}
        for k in FIELDS:
            entry[k] = _to_pos(lyr.get(k, EPS))
        # Helios convention: communication ~ gradient volume ~ parameter bytes
        entry["grad_bytes"] = entry["param_bytes"]
        clean_layers.append(entry)

    # Drop layers that are effectively empty across all fields (prevents near-zero blocks)
    pruned = []
    for e in clean_layers:
        s = (e["ops_fwd"] + e["ops_bwd"] + e["bytes_mv_fwd"] + e["bytes_mv_bwd"] + e["param_bytes"])
        if s <= 5 * EPS:
            continue
        pruned.append(e)

    if not pruned:
        raise ValueError(f"{path}: after sanitization, no valid layers remained")
    return pruned


def _split_indices(n: int, nblocks: int = 8) -> List[Tuple[int, int]]:
    base, rem = divmod(n, nblocks)
    parts = []
    i = 0
    for s in range(nblocks):
        take = base + (1 if s < rem else 0)
        parts.append((i, i + take))
        i += take
    return parts
def _layer_ops_and_bytes(lyr: Dict) -> Tuple[float, float]:
    """Per-layer totals for the 'diverse' splitter."""
    ops = float(lyr.get("ops_fwd", 0.0)) + float(lyr.get("ops_bwd", 0.0))
    b_f = float(lyr.get("bytes_mv_fwd", 0.0))
    b_b = float(lyr.get("bytes_mv_bwd", 0.0))
    p   = float(lyr.get("param_bytes", 0.0))
    g   = float(lyr.get("grad_bytes", p))
    by  = b_f + b_b + p + g
    return max(ops, EPS), max(by, EPS)


def _agg_range(layers: List[Dict], s: int, e_exclusive: int) -> Dict:
    """Sum the standard fields over [s, e) and floor to EPS."""
    sums = {
        "ops_fwd": 0.0, "ops_bwd": 0.0,
        "bytes_mv_fwd": 0.0, "bytes_mv_bwd": 0.0,
        "param_bytes": 0.0, "grad_bytes": 0.0
    }
    for i in range(s, e_exclusive):
        lyr = layers[i]
        for k in sums:
            sums[k] += _to_pos(lyr.get(k, EPS))
    return {k: max(float(v), EPS) for k, v in sums.items()}


def aggregate_blocks(layers: List[Dict], nblocks: int = 8) -> List[Dict]:
    """Sum sanitized per-layer metrics into 8 depth-contiguous blocks. Floors each field to EPS."""
    L = len(layers)
    if L < nblocks:
        # Very rare for LLMs; replicate to reach 8 blocks if needed
        layers = (layers * ((nblocks + L - 1) // L))[:nblocks]
        L = len(layers)

    blocks = []
    for s, e in _split_indices(L, nblocks):
        sums = {"ops_fwd": 0.0, "ops_bwd": 0.0, "bytes_mv_fwd": 0.0, "bytes_mv_bwd": 0.0, "param_bytes": 0.0, "grad_bytes": 0.0}
        for i in range(s, e):
            lyr = layers[i]
            for k in sums:
                sums[k] += _to_pos(lyr.get(k, EPS))
        # Floor to EPS so no zeros slip through
        blk = {k: max(float(v), EPS) for k, v in sums.items()}
        blocks.append(blk)
    return blocks
def aggregate_blocks_diverse(layers: List[Dict], nblocks: int = 8) -> List[Dict]:
    """
    Build nblocks contiguous blocks with diversity: even blocks target BYTES,
    odd blocks target OPS. This preserves serial order and produces blocks
    with different ops/bytes mixes without explicit classification.
    """
    L = len(layers)
    if L == 0:
        blk = {"ops_fwd": EPS, "ops_bwd": EPS,
               "bytes_mv_fwd": EPS, "bytes_mv_bwd": EPS,
               "param_bytes": EPS, "grad_bytes": EPS}
        return [dict(blk) for _ in range(nblocks)]

    # Precompute per-layer totals and global targets
    ops_list, bytes_list = [], []
    for lyr in layers:
        o, b = _layer_ops_and_bytes(lyr)
        ops_list.append(o)
        bytes_list.append(b)

    total_ops = sum(ops_list)
    total_bytes = sum(bytes_list)

    n_bytes_blocks = (nblocks + 1) // 2
    n_ops_blocks   = nblocks // 2
    bytes_target = total_bytes / max(n_bytes_blocks, 1)
    ops_target   = total_ops   / max(n_ops_blocks,   1)

    blocks = []
    s = 0
    for k in range(nblocks):
        if s >= L:
            last = blocks[-1] if blocks else _agg_range(layers, 0, min(1, L))
            blocks.append(dict(last))
            continue

        want_bytes = (k % 2 == 0)
        acc_ops = 0.0
        acc_bytes = 0.0
        e = s
        while e < L:
            acc_ops += ops_list[e]
            acc_bytes += bytes_list[e]
            e += 1

            remaining_layers = L - e
            remaining_blocks = nblocks - (k + 1)
            must_leave = remaining_layers < remaining_blocks
            target_hit = (acc_bytes >= bytes_target) if want_bytes else (acc_ops >= ops_target)

            if target_hit and not must_leave:
                break
            if must_leave and remaining_layers >= remaining_blocks:
                break

        blocks.append(_agg_range(layers, s, e))
        s = e

    if len(blocks) > nblocks:
        blocks = blocks[:nblocks]
    while len(blocks) < nblocks:
        blocks.append(dict(blocks[-1]))
    return blocks


def raw_summary_from_layers(layers: List[Dict]) -> Tuple[float, float, float, float]:
    """Return (depth_raw, comp_raw, mem_raw, comm_raw) as per-layer averages; each strictly >0."""
    L = max(len(layers), 1)
    total_ops = 0.0
    total_mem = 0.0
    total_comm = 0.0
    for lyr in layers:
        total_ops += _to_pos(lyr.get("ops_fwd", EPS)) + _to_pos(lyr.get("ops_bwd", EPS))
        total_mem += _to_pos(lyr.get("bytes_mv_fwd", EPS)) + _to_pos(lyr.get("bytes_mv_bwd", EPS))
        total_comm += _to_pos(lyr.get("param_bytes", EPS))  # comm ~ param_bytes
    depth_raw = float(L)
    comp_raw = total_ops / L
    mem_raw = total_mem / L
    comm_raw = total_comm / L
    return max(depth_raw, EPS), max(comp_raw, EPS), max(mem_raw, EPS), max(comm_raw, EPS)


def convert_folder(in_dir: str, out_dir: str, pattern: str = "*_bs4_sl256.pkl", segmentation: str = "diverse") -> None:
    os.makedirs(out_dir, exist_ok=True)
    src_paths = sorted(glob.glob(os.path.join(in_dir, pattern)))
    if not src_paths:
        raise FileNotFoundError(f"No PKLs matching {pattern!r} in {in_dir!r}")

    items = []
    max_depth = EPS
    max_comp = EPS
    max_mem = EPS
    max_comm = EPS

    for p in src_paths:
        name = os.path.splitext(os.path.basename(p))[0]
        layers = load_layers(p)

        if segmentation == "diverse":
            blocks = aggregate_blocks_diverse(layers, nblocks=8)
        elif segmentation == "equal":
            blocks = aggregate_blocks(layers, nblocks=8)
        else:
            blocks = aggregate_blocks_diverse(layers, nblocks=8)

        depth_raw, comp_raw, mem_raw, comm_raw = raw_summary_from_layers(layers)

        items.append(dict(name=name, layers=layers, blocks=blocks,
                          depth_raw=depth_raw, comp_raw=comp_raw, mem_raw=mem_raw, comm_raw=comm_raw))

        max_depth = max(max_depth, depth_raw)
        max_comp = max(max_comp, comp_raw)
        max_mem = max(max_mem, mem_raw)
        max_comm = max(max_comm, comm_raw)

    for it in items:
        name = it["name"]
        blocks_path = os.path.join(out_dir, f"{name}_blocks.pkl")
        with open(blocks_path, "wb") as f:
            pickle.dump({"blocks": it["blocks"]}, f)

        depth = float(it["depth_raw"] / max_depth)
        comp = float(it["comp_raw"] / max_comp)
        mem = float(it["mem_raw"] / max_mem)
        comm = float(it["comm_raw"] / max_comm)
        vec = np.array([depth, comp, mem, comm], dtype=np.float64)
        vec = np.clip(vec, EPS, 1.0).astype(np.float32)

        summary_path = os.path.join(out_dir, f"{name}_summary.pkl")
        with open(summary_path, "wb") as f:
            pickle.dump({"summary": vec}, f)

        print(f"[OK] wrote {blocks_path}  &  {summary_path}")

    bad = []
    for p in sorted(glob.glob(os.path.join(out_dir, "*_blocks.pkl"))):
        with open(p, "rb") as f:
            d = pickle.load(f)
        blks = d.get("blocks", [])
        if not isinstance(blks, list) or len(blks) != 8:
            bad.append((p, "blocks_count", len(blks)))
            continue
        for i, blk in enumerate(blks):
            for k, v in blk.items():
                v = float(v)
                if (not np.isfinite(v)) or (v <= 0.0):
                    bad.append((p, i, k, v))

    for p in sorted(glob.glob(os.path.join(out_dir, "*_summary.pkl"))):
        with open(p, "rb") as f:
            d = pickle.load(f)
        vec = np.array(d.get("summary", []), dtype=np.float64)
        if (vec.size != 4) or (not np.all(np.isfinite(vec))) or np.any(vec <= 0.0):
            bad.append((p, "summary_bad", vec.tolist()))

    if bad:
        print("\n[WARN] Some outputs failed the positivity/finite audit:")
        for b in bad:
            print("  ", b)
        raise SystemExit(2)
    else:
        print("\n[OK] All outputs passed audit: no zeros, no NaNs, finite values only.")



def main():
    ap = argparse.ArgumentParser(description="Convert LLM_profiler PKLs (bs4,sl256) to model_profiler blocks+summary.")
    ap.add_argument("--in", dest="in_dir", default="raw_profiles", help="Input folder with LLM_profiler PKLs")
    ap.add_argument("--out", dest="out_dir", default="profiles", help="Output folder for converted PKLs")
    ap.add_argument("--pattern", default="*_bs4_sl256.pkl", help="Glob to select input files")
    ap.add_argument("--segmentation", choices=["diverse", "equal"], default="diverse",
                    help="diverse: alternate bytes/ops targets (default); equal: depth-equal slices")
    args = ap.parse_args()
    convert_folder(args.in_dir, args.out_dir, args.pattern, segmentation=args.segmentation)



if __name__ == "__main__":
    main()