"""Keep your panel when your bot trains on: carry a thoughts pack over to a newer checkpoint of the same bot.

A pack belongs to one network: its features are numbers in one dictionary of one checkpoint. Train the bot further
and those numbers mean nothing in the new network - but most of the ideas are still there, and can be found again by
WHAT THEY FIRE ON. Both networks read the same recorded moments; each named feature of the old pack takes the name of
the new dictionary's feature whose activation goes up and down with it most closely (Pearson r over the moments). A
match below ``min_r`` is left out and reported: an idea that did not survive the training, or that the new
dictionary split in two. On a large self-play bot, 8 of 8 names carried over between checkpoints 10 billion steps
apart, r 0.65 to 0.96 (two with a runner-up close behind - worth a look).

    report = carry_names(old_pack, old_layer_values, new_features)       # the matching, one entry per name
    write_carried_names("names.toml", report, new_features)              # a names file for the new dictionary
    refit_pack("thoughts.npz", policy=new, old_policy=old, out="refit")  # all of it, the landing ring and the rest

``refit_pack`` (and ``cds-interpret refit-pack``) does every step for a new checkpoint: record games (or take yours),
a new dictionary, the names carried over, the probes refitted on their labels, the landing readout, calibrated
gauges, decoder directions and recorded moments for the two decision rows. Each step's result is kept in the output
folder, so running it again picks up where it stopped.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .errors import CDSInterpretError


def feature_activations(pack: dict[str, Any], h: Any) -> np.ndarray:
    """The pack's named features' activations [N, F] on a layer's values h [N, D] (the layer the pack reads): what
    the panel computes, max(w . h + b, 0) for each - every feature's own, without the dictionary's top-k."""
    feats = pack.get("features") or []
    if not feats:
        return np.zeros((len(h), 0), np.float32)
    w = np.stack([np.asarray(f["w"], np.float32) for f in feats], 1)
    b = np.array([float(f["b"]) for f in feats], np.float32)
    x = h.detach().cpu().float().numpy() if torch.is_tensor(h) else np.asarray(h, np.float32)
    if x.shape[1] != w.shape[0]:
        raise CDSInterpretError(f"the pack reads a layer of width {w.shape[0]}, these values have {x.shape[1]}: give "
                                "the OLD network's values of the pack's layer")
    return np.maximum(x @ w + b, 0.0)


def _code_moments(fs: Any, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    if getattr(fs, "codes", None) is None:
        raise CDSInterpretError("the new FeatureSet has no codes: make it with study.features(...) or "
                                "study.load_features(...)")
    idx = np.asarray(fs.codes[0].numpy() if torch.is_tensor(fs.codes[0]) else fs.codes[0], np.int64)
    val = np.asarray(fs.codes[1].float().numpy() if torch.is_tensor(fs.codes[1]) else fs.codes[1], np.float64)
    val = np.where(val > 0, val, 0.0)
    flat_i = idx.ravel()
    s1 = np.bincount(flat_i, val.ravel(), minlength=m)
    s2 = np.bincount(flat_i, (val * val).ravel(), minlength=m)
    return idx, val, s1, s2, len(idx)


def carry_names(old_pack: dict[str, Any], old_values: Any, new_fs: Any, min_r: float = 0.5) -> list[dict[str, Any]]:
    """Find each named feature of ``old_pack`` again in a new dictionary, by what it fires on.

    old_values [N, D]: the OLD network's values of the pack's layer on N recorded moments; new_fs: a FeatureSet of
    the NEW network (study.features), on the same N moments in the same order. For each name: the new feature whose
    activation correlates best with the old one (r), the runner-up (r2), and whether it is kept (r >= min_r, and no
    earlier name took that feature). [{"name", "old", "new", "r", "runner_up", "r2", "old_fires", "kept"}]"""
    za = feature_activations(old_pack, old_values).astype(np.float64)
    m = int(new_fs.sae.m)
    idx, val, s1b, s2b, n = _code_moments(new_fs, m)
    if len(za) != n:
        raise CDSInterpretError(f"{len(za)} moments of the old network's layer but {n} in the new dictionary's "
                                "codes: they must be the same moments, in the same order")
    ma = za.mean(0)
    sa = za.std(0)
    mb = s1b / n
    sb = np.sqrt(np.maximum(s2b / n - mb * mb, 0.0))
    out: list[dict[str, Any]] = []
    taken: set[int] = set()
    for f_i, f in enumerate(old_pack.get("features") or []):
        a = za[:, f_i]
        sab = np.bincount(idx.ravel(), (a[:, None] * val).ravel(), minlength=m) / n
        den = sa[f_i] * sb
        r = np.where(den > 0, (sab - ma[f_i] * mb) / np.where(den > 0, den, 1.0), 0.0)
        order = np.argsort(-r)
        best, second = int(order[0]), int(order[1]) if m > 1 else -1
        rb, r2 = float(r[best]), float(r[second]) if second >= 0 else float("nan")
        kept = bool(rb >= min_r and best not in taken)
        if kept:
            taken.add(best)
        out.append({"name": str(f["name"]), "old": int(f["index"]), "new": best, "r": round(rb, 3),
                    "runner_up": second, "r2": round(r2, 3), "old_fires": round(float((a > 0).mean()), 4),
                    "kept": kept})
    return out


def describe_carry(report: list[dict[str, Any]], min_r: float = 0.5) -> str:
    """The matching in words, one line a name."""
    lines = []
    for e in report:
        close = " - the runner-up is close: check its moments" if e["kept"] and e["r"] - e["r2"] < 0.1 else ""
        why = "" if e["kept"] else (f"  LEFT OUT (r under {min_r})" if e["r"] < min_r else "  LEFT OUT (taken)")
        lines.append(f"  {e['name']!r}: feature {e['old']} -> {e['new']}, r {e['r']:.2f} (runner-up {e['runner_up']} "
                     f"{e['r2']:.2f}){close}{why}")
    kept = sum(e["kept"] for e in report)
    return f"{kept} of {len(report)} names carried over:\n" + "\n".join(lines)


def write_carried_names(path: str | os.PathLike[str], report: list[dict[str, Any]], new_fs: Any,
                        source: str = "the old pack", overwrite: bool = False) -> Path:
    """A names file for the new dictionary: each carried name confirmed (they were confirmed before), with where it
    came from and how close the match was as a note; the names left out as comments, so you can look for them."""
    from .names import _q
    p = Path(path)
    if p.exists() and not overwrite:
        raise CDSInterpretError(f"{p} already exists: pass overwrite=True to replace it")
    lines = ["# Names carried over to a new dictionary by what the features fire on (cds_interpret.refit). Each was a",
             "# confirmed name in the old pack; look at the moments of any whose runner-up came close.",
             f"layer = {_q(new_fs.layer)}", f"dictionary = {_q(new_fs.fingerprint)}", ""]
    for e in report:
        if not e["kept"]:
            continue
        note = (f"carried over from feature {e['old']} of {source}: r {e['r']:.2f} (runner-up feature {e['runner_up']}, "
                f"r {e['r2']:.2f})")
        lines += [f"[features.{e['new']}]", f"name = {_q(e['name'])}", 'status = "confirmed"', f"note = {_q(note)}", ""]
    left = [e for e in report if not e["kept"]]
    if left:
        lines.append("# Not carried over (no feature of the new dictionary fires closely enough with them):")
        lines += [f"#   {e['name']!r} (old feature {e['old']}): best match feature {e['new']}, r {e['r']:.2f}"
                  for e in left]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _old_values(old_policy: Any, layer: str, study: Any) -> torch.Tensor:
    """The old network's values of its pack's layer on the study's moments: its own inputs from the game state when it
    has an obs function, else the moments' recorded inputs (the same layout as the new network's)."""
    x = old_policy.observe(study.moments.frames, study.moments.me) if old_policy.obs_fn is not None else study.x
    return old_policy.activations(x, [layer])[layer]


def refit_pack(old_pack: Any, policy: Any, old_policy: Any, out: str | os.PathLike[str], moments: Any = None,
               games: int = 24, seconds: float = 90.0, team_size: int = 1, opponent: Any = "chaser",
               layer: str | None = None, m: int | None = None, k: int | None = None, steps: int = 2000,
               min_r: float = 0.5, probes: dict[str, str] | None = None, donors: int = 32, calibrate: bool = True,
               log: Any = print) -> dict[str, Any]:
    """A thoughts pack for a newer checkpoint, every step, into the folder ``out``:

    1. moments: ``moments`` (a Moments, or a recording's folder) or ``games`` games recorded in RocketSim with the new
       policy (``out/recording``; needs its obs function and action table);
    2. a new dictionary of the pack's layer (``out/features.sae``; the old one's size unless m / k are given);
    3. the names carried over by what they fire on (``out/names.toml``; carry_names, ``min_r``);
    4. the pack: the probes refitted on their labels (packs from Study.make_pack remember them; ``probes`` {name:
       label} for others), the landing readout (Study.landing), decoder directions and ``donors`` recorded moments;
    5. its yes / no gauges calibrated on held-out games (Study.calibrate_pack);
    6. ``out/thoughts.npz`` and ``out/refit.json`` (every number of every step).

    old_pack: the old pack (a dict or its file); policy / old_policy: the new and old checkpoints (Policy objects or
    "module:factory"). Steps whose result is already in ``out`` are not done again. Returns the report."""
    from . import overlay as OV
    from .moments import Moments
    from .overlay.explain import load_policy
    from .record import rocketsim
    from .study import Study
    log = log or (lambda *_: None)
    outp = Path(out)
    outp.mkdir(parents=True, exist_ok=True)
    pk_old = old_pack if isinstance(old_pack, dict) else OV.load_pack(old_pack)
    source = "the old pack" if isinstance(old_pack, dict) else Path(old_pack).name
    pol, old = load_policy(policy), load_policy(old_policy)
    if pol is None or old is None:
        raise CDSInterpretError("refit_pack needs both checkpoints: policy= (the new one) and old_policy=")
    lay = layer or pk_old["layer"]
    report: dict[str, Any] = {"old_pack": source, "layer": lay}
    # 1. moments
    if moments is None:
        log(f"1/5 recording {games} games of the new checkpoint in RocketSim ({team_size}v{team_size}) ...")
        mo = rocketsim(pol, games=games, seconds=seconds, team_size=team_size, opponent=opponent,
                       out=outp / "recording", if_exists="reuse", log=log)
    elif isinstance(moments, Moments):
        mo = moments
    else:
        mo = Moments.load(moments) if (Path(moments) / "manifest.json").exists() else Moments.load_all(moments)
    if moments is not None:
        log(f"1/5 moments: {len(mo):,} given")
    report["moments"] = len(mo)
    study = Study(pol, mo, layers=None, log=log)
    if lay not in study.layers:
        study = Study(pol, mo, layers=study.layers + [lay], log=log)
    # 2. the new dictionary
    sae_path = outp / "features.sae"
    size = pk_old.get("meta", {}).get("sae") or {}
    if sae_path.exists():
        log(f"2/5 the new dictionary: {sae_path} (kept from an earlier run)")
        fs = study.load_features(sae_path, layer=lay)
    else:
        log(f"2/5 a new dictionary of {lay} ...")
        fs = study.features(lay, m=m or size.get("m"), k=k or size.get("k"), steps=steps)
        fs.save(sae_path)
    # 3. names
    log("3/5 carrying the names over by what they fire on ...")
    carried = carry_names(pk_old, _old_values(old, pk_old["layer"], study), fs, min_r)
    log(describe_carry(carried, min_r))
    names_path = write_carried_names(outp / "names.toml", carried, fs, source, overwrite=True)
    report["names"] = {"min_r": min_r, "features": carried}
    # 4. the pack
    want: dict[str, Any] = dict(probes or {})
    skipped = []
    for q in pk_old.get("probes") or []:
        if q["name"] in want:
            continue
        if q.get("label"):
            look = {k_: q[k_] for k_ in ("unit", "color") if q.get(k_) is not None}
            want[q["name"]] = (q["label"], look) if look else q["label"]
        else:
            skipped.append(q["name"])
    if skipped:
        log(f"  (probes without a label in the old pack, not refitted: {', '.join(skipped)} - give them as "
            "probes={name: label})")
    report["probes_skipped"] = skipped
    log("4/5 the pack: probes, landing ring, decoder directions, recorded moments ...")
    meta = pk_old.get("meta", {})
    kept = any(e["kept"] for e in carried)
    pack = study.make_pack(lay, names=names_path if kept else None, probes=want,
                           title=meta.get("title", "THOUGHTS"), value_scale=float(meta.get("value_scale", 1.0)),
                           output_layer=meta.get("output_layer", ""), donors=donors)
    report["landing"] = {kk: v for kk, v in (study.landing_results or {}).items() if kk not in ("w", "b")}
    # 5. calibration
    if calibrate:
        log("5/5 calibrating the yes / no gauges on held-out games ...")
        report["calibration"] = study.calibrate_pack(pack)
    path = OV.save_pack(pack, outp / "thoughts.npz")
    report["pack"] = Path(path).name
    (outp / "refit.json").write_text(json.dumps(report, indent=1, default=_jsonable) + "\n", encoding="utf-8")
    log(f"Wrote {path} - check it with: cds-interpret check-panel panel.toml {path}")
    return report


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return str(v)
