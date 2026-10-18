"""A thoughts pack: what the live overlay reads from one layer of your network, in one small file.

    pack = make_pack(
        layer="mlp.3",                                    # the module path, in the network your bot runs
        sae=sae, features={"chasing the ball": 12, "defending": 40},       # named SAE features
        probes={"Threat on goal": res["threat"]["probe"],                    # probes shown with a bar
                "Time to the ball": (res["ttb_me"]["probe"], {"unit": "s"})},
        actions=policy.actions, output=policy.output,     # to say what the policy leans toward
        title="MY BOT'S MIND", value_scale=1.0)
    save_pack(pack, "thoughts.npz")

What it holds:

- **features**: an SAE's encoder rows for the named features only (pre-activation = w . h + b in the layer's units)
  and a firing threshold each (:func:`cds_interpret.sae.firing_thresholds`: the feature's smallest typical firing
  value - the overlay computes a few features, not all of them, so it cannot run the SAE's top-k gate);
- **probes**: linear readouts of the layer - a yes / no probe is shown as a chance with a bar, a number with its
  unit (``"s"`` for seconds);
- **actions** and **output**: the action table and what the network outputs (logits, probs, log_probs), to show the
  move it leans toward and how sure it is;
- **outlook**: the outlook bar (green right: it expects to score; red left: in trouble) comes from your critic when
  the overlay is given one, else from a probe named in ``outlook_probe`` (fit one on the ``outlook`` label);
- **land** (optional, :func:`add_land`): the landing marker - where the network thinks the ball will come down -
  read from its own layer, often an earlier one;
- **decoders** (made by default, or :func:`add_decoders`): each named feature's decoder direction, for the
  "why this move" row - how hard each feature pushes the move the policy leans to;
- **donors** (optional, :func:`add_donors`): a few dozen real recorded moments, for the "looking at" row - what the
  move rests on: the bot's own car, the ball, the opponents, the team-mates.

Packs are saved as a plain ``.npz`` file (arrays and JSON text, no pickled code): safe to load from anyone.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from ..errors import CDSInterpretError, ShapeError
from ..frames import REQUIRED, Frames
from ..policy import describe_action

FORMAT = "cds-interpret.thoughts"
VERSION = 2                             # 2 (0.2): decoder directions and donor moments; 1 (0.1) still loads
DONOR_FIELDS = REQUIRED + ("prev_controls",)
DEFAULT_COLORS = [(0, 170, 255), (255, 196, 64), (90, 220, 120), (240, 90, 80), (200, 120, 255), (255, 140, 60)]


def _probe_parts(p: Any) -> tuple[np.ndarray, float, str]:
    """(w, b, kind) of a LinearProbe, a dict with w / b / kind, or a (w, b, kind) tuple."""
    if hasattr(p, "w") and hasattr(p, "b"):
        w = p.w.detach().cpu().numpy() if hasattr(p.w, "detach") else np.asarray(p.w)
        kind = str(getattr(p, "kind", "reg"))
        if kind == "bin" and getattr(p, "link", "identity") == "logit":
            kind = "bin_logit"
        return np.asarray(w, np.float32).reshape(-1), float(p.b), kind
    if isinstance(p, dict):
        return np.asarray(p["w"], np.float32).reshape(-1), float(p["b"]), str(p.get("kind", "reg"))
    w, b, kind = p
    return np.asarray(w, np.float32).reshape(-1), float(b), str(kind)


def make_pack(layer: str, sae: Any = None, features: dict[str, int] | None = None,
              thresholds: dict[Any, float] | None = None, probes: dict[str, Any] | None = None, actions: Any = None,
              output: str = "logits", output_layer: str = "", title: str = "THOUGHTS", value_scale: float = 1.0,
              outlook_probe: Any = None, meta: dict[str, Any] | None = None, decoders: bool = True) -> dict[str, Any]:
    """A pack for the module ``layer`` (its path in the network your bot runs; see the module's notes).

    Args:
        sae, features: the SAE of that layer and the features to show, {display name: feature index}.
        thresholds: firing thresholds, {display name or feature index: value} (default 0: on when positive).
        probes: {display name: probe}, a probe being a LinearProbe (probes.ridge_suite(..., keep=True)) or a
            ``(probe, look)`` pair, look = {"unit": "s" | "%" | "", "color": (r, g, b)}. Yes / no probes default
            to "%", number probes to "" (shown as the number).
        actions: [A, 8] the action table (to name the action the policy leans toward); output: what the network's
            output layer gives ("logits", "probs" or "log_probs"); output_layer: that module's path ("" = the
            whole network's output).
        title: the panel's title. value_scale: the outlook value that fills the bar.
        outlook_probe: a probe whose readout is the outlook when the overlay has no critic (e.g. one fitted on the
            ``outlook`` label, which runs from -1 to +1).
        decoders: keep each named feature's decoder direction too (a row of the layer's width each), for the
            panel's "why this move" row.
    """
    pack: dict[str, Any] = {"format": FORMAT, "version": VERSION, "layer": str(layer), "features": [], "probes": [],
                            "table": None if actions is None else np.asarray(actions, np.float32),
                            "meta": {"title": title, "value_scale": float(value_scale), "output": output,
                                     "output_layer": output_layer, **(meta or {})},
                            "outlook": None, "land": None}
    if pack["table"] is not None and (pack["table"].ndim != 2 or pack["table"].shape[1] != 8):
        raise ShapeError(f"actions must be [A, 8], not {pack['table'].shape}")
    if features and sae is None:
        raise CDSInterpretError("make_pack: features need the SAE they come from (sae=)")
    th = thresholds or {}
    for name, j in (features or {}).items():
        j = int(j)
        if not 0 <= j < sae.m:
            raise ShapeError(f"feature {name!r}: index {j} is outside the SAE's {sae.m} features")
        w = (sae.W_enc[:, j].detach().cpu().numpy() / float(sae.scale)).astype(np.float32)
        b = float(sae.b_enc[j].detach() - sae.b_pre.detach() @ sae.W_enc[:, j].detach())
        pack["features"].append({"name": str(name), "index": j, "w": w, "b": b,
                                 "threshold": float(th.get(name, th.get(j, 0.0)))})
        if decoders:
            pack["features"][-1]["d"] = _direction(sae, j)
    for i, (name, p) in enumerate((probes or {}).items()):
        look: dict[str, Any] = {}
        if isinstance(p, tuple) and len(p) == 2 and isinstance(p[1], dict):
            p, look = p
        w, b, kind = _probe_parts(p)
        unit = look.get("unit", "%" if kind.startswith("bin") else "")
        color = look.get("color", DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        pack["probes"].append({"name": str(name), "w": w, "b": b, "kind": kind, "unit": unit,
                               "color": [int(c) for c in color]})
    if outlook_probe is not None:
        w, b, kind = _probe_parts(outlook_probe)
        pack["outlook"] = {"w": w, "b": b}
    widths = {len(f["w"]) for f in pack["features"]} | {len(p["w"]) for p in pack["probes"]}
    if pack["outlook"] is not None:
        widths.add(len(pack["outlook"]["w"]))
    if len(widths) > 1:
        raise ShapeError(f"the features and probes read layers of different widths {sorted(widths)}: a pack reads ONE "
                         "layer (the landing marker can read another: add_land)")
    pack["meta"]["width"] = int(widths.pop()) if widths else None
    return pack


def add_land(pack: dict[str, Any], module: str, w: Any, b: Any, info: dict[str, Any] | None = None) -> dict[str, Any]:
    """The landing marker read from its own layer ``module``: (x, y) = h @ w + b with w [D, 2], b [2], in the
    bot's TEAM frame (the frame the labels use: it attacks +y). A place on the field often reads best - and far
    more steadily - from an EARLY layer: fit it with probes.ridge_suite on land_x / land_y there and check
    probes.readout_stability first (docs/howto/probes.md). While the bot plays, that layer is copied by one more
    read-only hook. info: anything to keep with it (how it was fitted, its held-out numbers)."""
    w = np.asarray(w, np.float32)
    b = np.asarray(b, np.float32).reshape(-1)
    if w.ndim != 2 or w.shape[1] != 2 or b.shape != (2,):
        raise ShapeError(f"add_land: w must be [D, 2] and b [2], not {w.shape} and {b.shape}")
    pack["land"] = {"module": str(module), "w": w, "b": b, "info": dict(info or {})}
    return pack


def _direction(sae: Any, j: int) -> np.ndarray:
    """Feature j's decoder direction in the layer's units [D] (what the feature adds to the layer per unit of its
    activation)."""
    d = sae.feature_direction(int(j)) if hasattr(sae, "feature_direction") else sae.W_dec[int(j)] * sae.scale
    return np.asarray(d.detach().float().cpu().numpy() if hasattr(d, "detach") else d, np.float32).reshape(-1)


def add_decoders(pack: dict[str, Any], sae: Any) -> dict[str, Any]:
    """Each named feature's decoder direction, from the SAE its features came from: what the panel's "why this move"
    row needs. A feature's push on the move the policy leans to is its activation x this direction x the gradient of
    that move's log-probability at the layer (attribution patching). Packs made by ``make_pack`` / ``Study.make_pack``
    have them already; this adds them to an older pack. Checks that the SAE is the features' own (their encoder rows
    must match)."""
    feats = pack.get("features") or []
    if not feats:
        raise CDSInterpretError("add_decoders: the pack has no named features, so there is nothing to add")
    for f in feats:
        j = int(f["index"])
        if not 0 <= j < int(sae.m):
            raise ShapeError(f"add_decoders: feature {f['name']!r} is number {j}, but this SAE has {sae.m} features: "
                             "it isn't the dictionary the pack's features came from")
        w = (sae.W_enc[:, j].detach().float().cpu().numpy() / float(sae.scale)).astype(np.float32)
        if w.shape != np.shape(f["w"]) or not np.allclose(w, f["w"], rtol=1e-4, atol=1e-6):
            raise CDSInterpretError(f"add_decoders: feature {f['name']!r} (number {j}) reads the layer differently in "
                                    "this SAE: it isn't the dictionary the pack's features came from. Pass that one "
                                    "(FeatureSet.save / Study.load_features keep it).")
        f["d"] = _direction(sae, j)
    pack["version"] = VERSION
    return pack


def _kickoff_rows(fr: Frames) -> np.ndarray:
    """[N] True where the ball sits still on the centre spot (a kickoff)."""
    pos, vel = np.asarray(fr.ball_pos, np.float64), np.asarray(fr.ball_vel, np.float64)
    return (np.abs(pos[:, 0]) < 1.0) & (np.abs(pos[:, 1]) < 1.0) & (np.linalg.norm(vel, axis=1) < 1.0)


def add_donors(pack: dict[str, Any], moments: Any, me: Any = None, n: int = 32, seed: int = 0,
               skip_kickoffs: bool = True) -> dict[str, Any]:
    """Real recorded moments for the panel's "looking at" row. While the bot plays, each part of the live moment -
    the bot's own car, the ball, the opponents, the team-mates - is swapped for the same part of one of these moments,
    and the move's probability over every combination says how much the move rests on each part.

    moments: a Moments (its frames and the car each moment was seen by), a list of them (one per mode: 1v1, 2v2,
    3v3), or Frames with ``me`` [N]. Up to ``n`` moments are kept per number of cars (kickoffs, and moments whose own
    car is demolished, left out), replacing any the pack had for that number of cars. Only the game state is kept
    (positions, speeds, boost, the cars' last controls when recorded) - a few kilobytes, and nothing of your network:
    your bot's input is rebuilt from them with its own obs function, in the overlay's worker
    (``with_thoughts(..., policy=...)``)."""
    if isinstance(moments, (list, tuple)):
        for m in moments:
            add_donors(pack, m, None, n, seed, skip_kickoffs)
        return pack
    fr = moments if isinstance(moments, Frames) else getattr(moments, "frames", None)
    if not isinstance(fr, Frames):
        raise CDSInterpretError(f"add_donors: give a Moments, a list of them, or Frames with me=, not "
                                f"{type(moments).__name__}")
    me = getattr(moments, "me", None) if me is None else me
    if me is None:
        raise CDSInterpretError("add_donors: with Frames, say which car each moment is seen by: me=[N] car slots")
    me = np.broadcast_to(np.asarray(me, np.int64), (len(fr),))
    rows = np.arange(len(fr))
    keep = np.ones(len(fr), bool)
    if skip_kickoffs:
        keep &= ~_kickoff_rows(fr)
    keep &= np.asarray(fr.car_demoed, np.float64)[rows, me] < 0.5
    keep &= np.asarray(fr.car_present, np.float64)[rows, me] > 0.5
    cand = rows[keep]
    if not len(cand):
        raise CDSInterpretError("add_donors: no usable moments (every one is a kickoff, or its own car is demolished)")
    pick = np.sort(np.random.default_rng(seed).choice(cand, size=min(int(n), len(cand)), replace=False))
    data = {k: np.asarray(fr.data[k][pick], np.float32) for k in DONOR_FIELDS if k in fr.data}
    group = {"cars": int(fr.n_cars), "me": me[pick].astype(np.int64), "data": data}
    donors = [g for g in pack.get("donors") or [] if int(g["cars"]) != group["cars"]]
    pack["donors"] = sorted(donors + [group], key=lambda g: int(g["cars"]))
    pack["version"] = VERSION
    return pack


# ---------------------------------------------------------------------------------------------------- files
def save_pack(pack: dict[str, Any], path: str | os.PathLike[str]) -> str:
    """Write a pack as one .npz file (arrays and JSON text only)."""
    arrays: dict[str, np.ndarray] = {}
    meta = {k: v for k, v in pack.items() if k not in ("features", "probes", "table", "outlook", "land", "donors")}
    meta["features"] = [{k: v for k, v in f.items() if k not in ("w", "d")} for f in pack["features"]]
    meta["probes"] = [{k: v for k, v in p.items() if k != "w"} for p in pack["probes"]]
    for i, f in enumerate(pack["features"]):
        arrays[f"feature_{i}"] = np.asarray(f["w"], np.float32)
        if f.get("d") is not None:
            arrays[f"decoder_{i}"] = np.asarray(f["d"], np.float32)
    for i, p in enumerate(pack["probes"]):
        arrays[f"probe_{i}"] = np.asarray(p["w"], np.float32)
    if pack.get("table") is not None:
        arrays["table"] = np.asarray(pack["table"], np.float32)
    if pack.get("outlook") is not None:
        arrays["outlook_w"] = np.asarray(pack["outlook"]["w"], np.float32)
        meta["outlook_b"] = float(pack["outlook"]["b"])
    if pack.get("land") is not None:
        arrays["land_w"] = np.asarray(pack["land"]["w"], np.float32)
        arrays["land_b"] = np.asarray(pack["land"]["b"], np.float32)
        meta["land"] = {"module": pack["land"]["module"], "info": pack["land"].get("info", {})}
    if pack.get("donors"):
        meta["donors"] = []
        for g, grp in enumerate(pack["donors"]):
            arrays[f"donors{g}_me"] = np.asarray(grp["me"], np.int64)
            for k, v in grp["data"].items():
                arrays[f"donors{g}_{k}"] = np.asarray(v, np.float32)
            meta["donors"].append({"cars": int(grp["cars"]), "n": int(len(grp["me"])), "fields": sorted(grp["data"])})
    arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    path = os.fspath(path)
    if not path.endswith(".npz"):
        path += ".npz"
    np.savez(path, **arrays)  # type: ignore[arg-type]
    return path


def load_pack(path: str | os.PathLike[str]) -> dict[str, Any]:
    """A pack written by save_pack (no pickles: allow_pickle stays off)."""
    with np.load(os.fspath(path), allow_pickle=False) as z:
        meta = json.loads(bytes(z["meta_json"]).decode("utf-8"))
        if meta.get("format") != FORMAT:
            raise CDSInterpretError(f"{path} is not a cds-interpret thoughts pack")
        pack = {k: v for k, v in meta.items() if k not in ("features", "probes", "outlook_b", "land", "donors")}
        pack["features"] = [{**f, "w": z[f"feature_{i}"], **({"d": z[f"decoder_{i}"]} if f"decoder_{i}" in z else {})}
                            for i, f in enumerate(meta["features"])]
        pack["probes"] = [{**p, "w": z[f"probe_{i}"]} for i, p in enumerate(meta["probes"])]
        pack["table"] = z.get("table")
        pack["outlook"] = {"w": z["outlook_w"], "b": meta["outlook_b"]} if "outlook_w" in z else None
        pack["land"] = ({"module": meta["land"]["module"], "w": z["land_w"], "b": z["land_b"],
                         "info": meta["land"].get("info", {})} if "land_w" in z else None)
        pack["donors"] = [{"cars": int(g["cars"]), "me": z[f"donors{i}_me"],
                           "data": {k: z[f"donors{i}_{k}"] for k in g["fields"]}}
                          for i, g in enumerate(meta.get("donors") or [])]
    return pack


# ---------------------------------------------------------------------------------------------- reading a layer
class Thoughts:
    """Reads a pack from a layer's values: a few dot products of the layer's width, pure numpy."""

    def __init__(self, pack: dict[str, Any]):
        self.pack = pack
        fs, ps = pack["features"], pack["probes"]
        self.fw = np.stack([f["w"] for f in fs]).astype(np.float32) if fs else np.zeros((0, 1), np.float32)
        self.fb = np.array([f["b"] for f in fs], np.float32)
        self.th = np.array([f["threshold"] for f in fs], np.float32)
        self.pw = np.stack([p["w"] for p in ps]).astype(np.float32) if ps else np.zeros((0, 1), np.float32)
        self.pb = np.array([p["b"] for p in ps], np.float32)
        self.table = pack.get("table")
        self.output = pack.get("meta", {}).get("output", "logits")
        ol = pack.get("outlook")
        self.outlook = None if ol is None else (np.asarray(ol["w"], np.float32), float(ol["b"]))
        land = pack.get("land")
        self.land_module = None if land is None else land["module"]
        self.land = None if land is None else (np.asarray(land["w"], np.float32), np.asarray(land["b"], np.float32))
        # "why this move": the named features' decoder directions [F, D] (None when the pack has none)
        self.fd = (np.stack([np.asarray(f["d"], np.float32) for f in fs])
                   if fs and all(f.get("d") is not None for f in fs) else None)
        # "looking at": real recorded moments, one group per number of cars: [(Frames, me [n])]
        self.donors: list[tuple[Frames, np.ndarray]] = []
        for g in pack.get("donors") or []:
            data = {k: np.asarray(v) for k, v in g["data"].items()}
            if all(k in data for k in REQUIRED) and len(g["me"]):
                self.donors.append((Frames(data), np.asarray(g["me"], np.int64)))

    @property
    def layer(self) -> str:
        return self.pack["layer"]

    @property
    def names(self) -> list[str]:
        """The named features, in the pack's order."""
        return [str(f["name"]) for f in self.pack["features"]]

    def donor_cars(self) -> list[int]:
        """The numbers of cars the pack has donor moments for (2 in 1v1, 4 in 2v2, 6 in 3v3)."""
        return [fr.n_cars for fr, _ in self.donors]

    def land_xy(self, h: Any) -> list[float] | None:
        """The landing marker (x, y; the team's frame) from its layer's values, or None without one."""
        if self.land is None:
            return None
        return [float(v) for v in np.asarray(h, np.float32).reshape(-1) @ self.land[0] + self.land[1]]

    def outlook_value(self, h: Any) -> float | None:
        if self.outlook is None:
            return None
        return float(np.asarray(h, np.float32).reshape(-1) @ self.outlook[0] + self.outlook[1])

    def read(self, h: Any, output: Any = None) -> dict[str, Any]:
        """What to draw for one car: {"features": [{"name", "act", "on"}], "probes": {name: value},
        "looks": {name: {"unit", "color"}}, "action": {"index", "p", "words"} | None, "value": outlook or None}."""
        h = np.asarray(h, np.float32).reshape(-1)
        out: dict[str, Any] = {"features": [], "probes": {}, "looks": {}, "action": None,
                               "value": self.outlook_value(h)}
        if len(self.fb):
            if self.fw.shape[1] != len(h):
                raise ShapeError(f"the pack reads a layer of width {self.fw.shape[1]}, the layer has {len(h)}")
            pre = self.fw @ h + self.fb
            for f, a, t in zip(self.pack["features"], pre, self.th):
                out["features"].append({"name": f["name"], "act": float(max(a, 0.0)), "on": bool(a > max(t, 0.0))})
        if len(self.pb):
            sc = self.pw @ h + self.pb
            for p, s in zip(self.pack["probes"], sc):
                kind = p["kind"]
                if kind == "bin_logit":
                    v = float(1.0 / (1.0 + np.exp(-s)))
                elif kind == "bin":
                    v = float(np.clip(s, 0.0, 1.0))
                else:
                    v = float(s)
                out["probes"][p["name"]] = v
                out["looks"][p["name"]] = {"unit": p.get("unit", ""), "color": p.get("color")}
        if output is not None:
            o = np.asarray(output, np.float64).reshape(-1)
            if self.output == "probs":
                pr = np.clip(o, 0, None)
            else:                                   # logits or log-probabilities: a softmax gives the probabilities
                pr = np.exp(o - o.max())
            pr = pr / max(pr.sum(), 1e-12)
            a = int(pr.argmax())
            words = describe_action(self.table[a]) if self.table is not None and a < len(self.table) else f"action {a}"
            out["action"] = {"index": a, "p": float(pr[a]), "words": words}
        return out
