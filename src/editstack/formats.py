"""Read and write model weights in several formats.

Every tensor is presented to edits as a numpy array. Storage dtypes numpy lacks (bfloat16) are exposed as float32,
and on write each tensor is cast back to its original storage dtype, so unchanged tensors come out bit-identical.

Supported (read + write):
  safetensors   single file or sharded folder (model-0000x-of-0000y.safetensors + index); native, no extra deps
  npz           numpy archive
  textdump      '## <name> <dtype> shape=[...]' blocks, one row per line (the astra model.txt format)
  torch         .pt/.pth/.bin/.ckpt state_dict, or TorchScript (needs `pip install torch`)
  onnx          initializers of an .onnx graph (needs `pip install onnx`)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------------------------- dtype helpers

_ST2NP = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U64": np.uint64, "U32": np.uint32, "U16": np.uint16, "U8": np.uint8, "BOOL": np.bool_,
}
_NP2ST = {np.dtype(v).str: k for k, v in _ST2NP.items()}


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 -> bfloat16 bits (uint16). NaN stays NaN."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    nan = np.isnan(x)
    if nan.any():
        r[nan] = 0x7FC0
    return r


def to_storage(arr: np.ndarray, dtype: str) -> np.ndarray:
    """Cast an edited array back to the checkpoint's storage dtype (a safetensors dtype tag)."""
    if dtype == "BF16":
        return f32_to_bf16(arr)
    return np.ascontiguousarray(arr, dtype=_ST2NP[dtype])


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------- checkpoint


@dataclass
class Checkpoint:
    """All tensors of one model plus enough format information to write it back."""
    fmt: str
    source: Path
    tensors: Dict[str, np.ndarray]           # editable view: float32 for bf16, native dtype otherwise
    dtypes: Dict[str, str]                   # storage dtype tag per tensor (safetensors naming)
    shard_of: Dict[str, str] = field(default_factory=dict)   # tensor -> shard file name (sharded safetensors)
    metadata: Dict[str, str] = field(default_factory=dict)
    extra_files: List[Path] = field(default_factory=list)    # config/tokenizer/normalizer files to carry along
    weight_files: List[Path] = field(default_factory=list)

    def names(self) -> List[str]:
        return list(self.tensors)


WEIGHT_SUFFIXES = {".safetensors", ".npz", ".pt", ".pth", ".bin", ".ckpt", ".onnx", ".txt"}


def detect(path: Path) -> str:
    path = Path(path)
    if path.is_dir():
        files = [p for p in path.iterdir() if p.is_file()]
        sts = [p for p in files if p.suffix == ".safetensors"]
        if sts:
            return "safetensors"
        for suf, fmt in ((".npz", "npz"), (".onnx", "onnx"), (".pt", "torch"), (".pth", "torch"),
                         (".bin", "torch"), (".ckpt", "torch")):
            if any(p.suffix == suf for p in files):
                return fmt
        if any(p.suffix == ".txt" and _is_textdump(p) for p in files):
            return "textdump"
        raise ValueError(f"no recognised weight files in {path}")
    suf = path.suffix
    if suf == ".safetensors":
        return "safetensors"
    if suf == ".npz":
        return "npz"
    if suf == ".onnx":
        return "onnx"
    if suf in (".pt", ".pth", ".bin", ".ckpt"):
        return "torch"
    if _is_textdump(path):
        return "textdump"
    raise ValueError(f"unknown weight format: {path}")


def _is_textdump(p: Path) -> bool:
    try:
        with open(p) as f:
            for _ in range(20):
                line = f.readline()
                if line.startswith("## ") and "shape=[" in line:
                    return True
    except (OSError, UnicodeDecodeError):
        return False
    return False


def _weight_files(path: Path, fmt: str) -> List[Path]:
    if path.is_file():
        return [path]
    files = sorted(p for p in path.iterdir() if p.is_file())
    if fmt == "safetensors":
        # weights only; small normalizer/processor safetensors are carried as extra files if a main model file exists
        main = [p for p in files if p.suffix == ".safetensors" and (p.name.startswith("model") or p.name.startswith("pytorch_model"))]
        return main or [p for p in files if p.suffix == ".safetensors"]
    if fmt == "textdump":
        return [p for p in files if p.suffix == ".txt" and _is_textdump(p)]
    suffix = {"npz": {".npz"}, "onnx": {".onnx"}, "torch": {".pt", ".pth", ".bin", ".ckpt"}}[fmt]
    return [p for p in files if p.suffix in suffix]


def load(path: Path, fmt: Optional[str] = None) -> Checkpoint:
    path = Path(path).expanduser().resolve()
    fmt = fmt or detect(path)
    files = _weight_files(path, fmt)
    if not files:
        raise ValueError(f"no {fmt} weight files in {path}")
    ck = Checkpoint(fmt=fmt, source=path, tensors={}, dtypes={}, weight_files=files)
    for f in files:
        {"safetensors": _load_st, "npz": _load_npz, "textdump": _load_text,
         "torch": _load_torch, "onnx": _load_onnx}[fmt](f, ck)
    if path.is_dir():
        ck.extra_files = sorted(p for p in path.iterdir() if p.is_file() and p not in files and not p.name.startswith("."))
    return ck


def save(ck: Checkpoint, tensors: Dict[str, np.ndarray], out_dir: Path, copy_extra: bool = True) -> List[Path]:
    """Write `tensors` (same names/shapes as ck) into out_dir in ck's format. Returns written weight files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = set(ck.tensors) - set(tensors)
    extra = set(tensors) - set(ck.tensors)
    if missing or extra:
        raise ValueError(f"tensor set changed: missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")
    for k, v in tensors.items():
        if tuple(v.shape) != tuple(ck.tensors[k].shape):
            raise ValueError(f"shape of {k} changed: {ck.tensors[k].shape} -> {v.shape}")
    written = {"safetensors": _save_st, "npz": _save_npz, "textdump": _save_text,
               "torch": _save_torch, "onnx": _save_onnx}[ck.fmt](ck, tensors, out_dir)
    if copy_extra:
        import shutil
        for f in ck.extra_files:
            shutil.copy2(f, out_dir / f.name)
    return written


# ---------------------------------------------------------------------------------------------- safetensors


def read_st_header(f: Path):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    return header, 8 + n


def _load_st(f: Path, ck: Checkpoint) -> None:
    header, base = read_st_header(f)
    meta = header.pop("__metadata__", None) or {}
    ck.metadata.update(meta)
    mm = np.memmap(f, dtype=np.uint8, mode="r")
    for name, info in header.items():
        dt, shape, (a, b) = info["dtype"], info["shape"], info["data_offsets"]
        raw = mm[base + a: base + b]
        if dt == "BF16":
            arr = bf16_to_f32(raw.view(np.uint16)).reshape(shape)
        elif dt in _ST2NP:
            arr = raw.view(_ST2NP[dt]).reshape(shape)
        else:
            raise ValueError(f"{f.name}:{name}: unsupported safetensors dtype {dt}")
        ck.tensors[name] = arr
        ck.dtypes[name] = dt
        ck.shard_of[name] = f.name


def write_safetensors(path: Path, arrays: Dict[str, np.ndarray], dtypes: Dict[str, str], metadata=None) -> None:
    header, offset, blobs = {}, 0, []
    for name in arrays:   # keep caller order
        data = to_storage(arrays[name], dtypes[name])
        nb = data.nbytes
        header[name] = {"dtype": dtypes[name], "shape": list(arrays[name].shape), "data_offsets": [offset, offset + nb]}
        blobs.append(data)
        offset += nb
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        for d in blobs:
            fh.write(memoryview(np.ascontiguousarray(d)).cast("B"))
    os.replace(tmp, path)


def _save_st(ck: Checkpoint, tensors, out_dir: Path) -> List[Path]:
    out = []
    shards = sorted(set(ck.shard_of.values()))
    for shard in shards:
        names = [n for n in ck.tensors if ck.shard_of[n] == shard]
        p = out_dir / shard
        write_safetensors(p, {n: tensors[n] for n in names}, ck.dtypes, ck.metadata or None)
        out.append(p)
    return out


# ---------------------------------------------------------------------------------------------- npz


def _load_npz(f: Path, ck: Checkpoint) -> None:
    z = np.load(f, allow_pickle=False)
    for k in z.files:
        a = z[k]
        ck.tensors[k] = a
        ck.dtypes[k] = _NP2ST.get(a.dtype.str, "F32")
        ck.shard_of[k] = f.name


def _save_npz(ck, tensors, out_dir: Path):
    out = []
    for shard in sorted(set(ck.shard_of.values())):
        p = out_dir / shard
        np.savez(p, **{n: to_storage(tensors[n], ck.dtypes[n]) for n in ck.tensors if ck.shard_of[n] == shard})
        out.append(p)
    return out


# ---------------------------------------------------------------------------------------------- text dump

_HDR = re.compile(r"^## (\S+) (\S+) shape=\[([0-9, ]*)\]")


def _load_text(f: Path, ck: Checkpoint) -> None:
    """Streams the file; each tensor is filled into a preallocated float32 buffer (multi-GB dumps are fine)."""
    name, dt, shape, buf, pos = None, None, None, None, 0

    def flush():
        if name is None:
            return
        if pos != buf.size:
            raise ValueError(f"{f.name}:{name}: expected {buf.size} values, read {pos}")
        ck.tensors[name] = buf.reshape(shape)
        ck.dtypes[name] = "F32"
        ck.metadata.setdefault("textdump_dtype:" + name, dt)
        ck.shard_of[name] = f.name

    with open(f) as fh:
        for line in fh:
            if line.startswith("#"):
                m = _HDR.match(line)
                if m:
                    flush()
                    name, dt = m.group(1), m.group(2)
                    shape = [int(s) for s in m.group(3).replace(" ", "").split(",") if s]
                    buf, pos = np.empty(int(np.prod(shape)) if shape else 1, np.float32), 0
                continue
            if name is not None and line.strip():
                vals = np.array(line.split(), dtype=np.float32)
                buf[pos: pos + vals.size] = vals
                pos += vals.size
    flush()


def _save_text(ck, tensors, out_dir: Path):
    out = []
    for shard in sorted(set(ck.shard_of.values())):
        p = out_dir / shard
        with open(p, "w") as fh:
            fh.write("# float32, %.9g\n# Each block: '## <name> <dtype> shape=[...]' then one row per line (dim 0), values space-separated\n")
            for n in ck.tensors:
                if ck.shard_of[n] != shard:
                    continue
                a = np.asarray(tensors[n], dtype=np.float32)
                dt = ck.metadata.get("textdump_dtype:" + n, "F32")
                fh.write(f"\n## {n} {dt} shape=[{', '.join(map(str, a.shape))}]\n")
                rows = a.reshape(a.shape[0], -1) if a.ndim > 1 else a.reshape(1, -1)
                for r in rows:
                    fh.write(" ".join(f"{v:.9g}" for v in r) + "\n")
        out.append(p)
    return out


# ---------------------------------------------------------------------------------------------- torch


def _torch():
    try:
        import torch  # noqa
        return torch
    except ImportError as e:  # pragma: no cover
        raise SystemExit("this checkpoint is a torch file: install torch in this environment (pip install torch)") from e


def _load_torch(f: Path, ck: Checkpoint) -> None:
    torch = _torch()
    try:
        obj = torch.load(f, map_location="cpu", weights_only=True)
        kind = "state_dict"
    except Exception:
        obj = torch.jit.load(str(f), map_location="cpu").state_dict()
        kind = "torchscript"
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
        kind = "nested_state_dict"
    ck.metadata["torch_kind:" + f.name] = kind
    for k, v in obj.items():
        if not hasattr(v, "dtype"):
            continue
        if v.dtype == torch.bfloat16:
            ck.tensors[k], ck.dtypes[k] = v.float().numpy(), "BF16"
        else:
            a = v.detach().cpu().numpy()
            ck.tensors[k], ck.dtypes[k] = a, _NP2ST.get(a.dtype.str, "F32")
        ck.shard_of[k] = f.name


def _save_torch(ck, tensors, out_dir: Path):
    torch = _torch()
    out = []
    for shard in sorted(set(ck.shard_of.values())):
        names = [n for n in ck.tensors if ck.shard_of[n] == shard]
        kind = ck.metadata.get("torch_kind:" + shard, "state_dict")

        def tt(n):
            if ck.dtypes[n] == "BF16":
                return torch.from_numpy(np.ascontiguousarray(tensors[n], np.float32)).to(torch.bfloat16)
            return torch.from_numpy(to_storage(tensors[n], ck.dtypes[n]).copy())

        p = out_dir / shard
        if kind == "torchscript":
            m = torch.jit.load(str(ck.source / shard if ck.source.is_dir() else ck.source), map_location="cpu")
            sd = m.state_dict()
            with torch.no_grad():
                for n in names:
                    sd[n].copy_(tt(n))
            m.save(str(p))
        else:
            sd = {n: tt(n) for n in names}
            torch.save({"state_dict": sd} if kind == "nested_state_dict" else sd, p)
        out.append(p)
    return out


# ---------------------------------------------------------------------------------------------- onnx


def _onnx():
    try:
        import onnx  # noqa
        from onnx import numpy_helper  # noqa
        return onnx, numpy_helper
    except ImportError as e:  # pragma: no cover
        raise SystemExit("this checkpoint is ONNX: install onnx (pip install onnx)") from e


def _load_onnx(f: Path, ck: Checkpoint) -> None:
    onnx, nh = _onnx()
    m = onnx.load(str(f))
    for init in m.graph.initializer:
        a = nh.to_array(init)
        if a.dtype.kind not in "fiub":
            continue
        ck.tensors[init.name] = a
        ck.dtypes[init.name] = _NP2ST.get(a.dtype.str, "F32")
        ck.shard_of[init.name] = f.name


def _save_onnx(ck, tensors, out_dir: Path):
    onnx, nh = _onnx()
    out = []
    for shard in sorted(set(ck.shard_of.values())):
        m = onnx.load(str(ck.source / shard if ck.source.is_dir() else ck.source))
        for i, init in enumerate(m.graph.initializer):
            if init.name in tensors and ck.shard_of.get(init.name) == shard:
                new = nh.from_array(to_storage(tensors[init.name], ck.dtypes[init.name]), init.name)
                m.graph.initializer[i].CopyFrom(new)
        p = out_dir / shard
        onnx.save(m, str(p))
        out.append(p)
    return out
