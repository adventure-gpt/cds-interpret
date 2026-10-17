"""How good is a feature dictionary? Numbers to compare SAEs (variants, seeds, sizes) beyond its reconstruction.

    card = scorecard(sae, x_eval, downstream, idx, val, labels, kinds)   # one dict of everything below
    match = decoder_match(sae_a, sae_b)                                  # per feature of a: best cosine in b
    stab = stability(sae_a, sae_b, freq_a)                               # share of a's features (and firing) b has
    al = label_alignment(idx, val, labels, kinds, m)                     # how well single features detect labels

- faithfulness to the POLICY (sae.downstream_check): KL from the real policy, argmax changed, KL recovered against
  mean ablation, FVU, L0. For a policy network this matters more than FVU: the 10-16 a.h3 dictionary explained 86%
  of the variance and still moved a third of the argmax choices.
- liveness: dead (never fire on the evaluation rows) and dense (fire on more than 10% of rows) features.
- stability (decoder_match / stability, activation_match): train twice with different seeds; a feature is
  "reproduced" when the other dictionary has a decoder direction with cosine >= 0.7 (also reported at 0.9), or - the
  better test - a feature whose activation correlates >= 0.7 with it on the same rows. Names given to features of
  one dictionary are only as solid as this. Measured on a 31M-weight bot's layer (8,192 features, k 32): two seeds
  shared 2% of features by decoder cosine >= 0.7 but 34% by activation r >= 0.7 (median best r 0.61), and a named
  "about to demo" feature had no partner above r 0.60.
- label alignment (label_alignment): for every binary label the single best feature's held-out AUROC and average
  precision ("1-sparse probing"), and the share of live features that are "label-explained" (a binary label with
  lift >= 5 and recall >= 0.05, or a regression label with |r| >= 0.5): how much of the dictionary our concept
  labels can name at all. Low is not bad per se - it says where the labels run out.
"""
from __future__ import annotations

import numpy as np
import torch

from .analysis import sparse_columns, sparse_relations
from .probes import average_precision


@torch.no_grad()
def decoder_match(a, b, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """For each feature of dictionary a: (the best cosine between its decoder direction and any of b's, its index)."""
    Wa = a.W_dec.detach().float()
    Wb = b.W_dec.detach().float()
    Wa = Wa / Wa.norm(dim=1, keepdim=True).clamp_min(1e-8)
    Wb = Wb / Wb.norm(dim=1, keepdim=True).clamp_min(1e-8)
    best, idx = [], []
    for i in range(0, len(Wa), chunk):
        c = Wa[i:i + chunk] @ Wb.T
        v, j = c.max(1)
        best.append(v.cpu())
        idx.append(j.cpu())
    return torch.cat(best).numpy(), torch.cat(idx).numpy()


def stability(a, b, freq_a: np.ndarray | None = None, live_min: float = 2e-4) -> dict:
    """Share of a's live features (firing on >= live_min of rows when freq_a is given) with a match in b at cosine
    >= 0.7 / 0.9, plain and weighted by firing frequency; and the share of mutual best matches."""
    ca, ia = decoder_match(a, b)
    _, ib = decoder_match(b, a)
    live = np.ones(len(ca), bool) if freq_a is None else np.asarray(freq_a) >= live_min
    w = np.ones(len(ca)) if freq_a is None else np.asarray(freq_a, np.float64)
    mutual = ib[ia] == np.arange(len(ia))
    out = {"n_live": int(live.sum()), "median_best_cos": float(np.median(ca[live]))}
    for t in (0.5, 0.7, 0.9):
        ok = (ca >= t) & live
        out[f"share_cos_{t}"] = float(int(ok.sum()) / max(int(live.sum()), 1))
        out[f"firing_weighted_cos_{t}"] = float(float((w * ok).sum()) / max(float((w * live).sum()), 1e-12))
    out["share_mutual_best"] = float(int((mutual & live).sum()) / max(int(live.sum()), 1))
    return {"summary": out, "best_cos": ca, "best_idx": ia}


def label_alignment(idx: np.ndarray, val: np.ndarray, labels: dict[str, np.ndarray], kinds: dict[str, str], m: int,
                    min_freq: float = 2e-4, top_for_ap: int = 5) -> dict:
    """Single-feature detection of each label and the share of live features some label explains (module notes).
    idx / val [n, k] the compact codes of the evaluation rows; labels name -> [n] (NaN = undefined)."""
    n = len(idx)
    r_nz, v_nz, st = sparse_columns(idx, val, m)
    freq = np.diff(st) / n
    live = np.nonzero(freq >= min_freq)[0]
    rel = {}
    for j in live:
        rel[int(j)] = sparse_relations(r_nz[st[j]:st[j + 1]], v_nz[st[j]:st[j + 1]], labels, kinds, n)
    per_label = {}
    for lab, kind in kinds.items():
        if lab not in labels:
            continue
        if kind == "bin":
            cand = sorted(((r[lab]["auroc"], j) for j, r in rel.items() if lab in r), reverse=True)
            if not cand:
                continue
            y = np.asarray(labels[lab], np.float64)
            ok = ~np.isnan(y)
            aps = []
            for _, j in cand[:top_for_ap]:
                a = np.zeros(n, np.float32)
                a[r_nz[st[j]:st[j + 1]]] = v_nz[st[j]:st[j + 1]]
                aps.append((average_precision(a[ok], y[ok]), j))
            best_ap, j_ap = max(aps)
            per_label[lab] = {"best_auroc": float(cand[0][0]), "feature_auroc": int(cand[0][1]),
                              "best_ap": float(best_ap), "feature_ap": int(j_ap),
                              "base": float((y[ok] > 0.5).mean())}
        else:
            cand = sorted(((abs(r[lab]["r"]), j) for j, r in rel.items() if lab in r), reverse=True)
            if cand:
                per_label[lab] = {"best_abs_r": float(cand[0][0]), "feature": int(cand[0][1])}
    explained = 0
    for r in rel.values():
        hit = any((("lift" in v and np.isfinite(v["lift"]) and v["lift"] >= 5 and v["recall"] >= 0.05)
                   or ("r" in v and abs(v["r"]) >= 0.5)) for v in r.values())
        explained += bool(hit)
    bins = [v for v in per_label.values() if "best_auroc" in v]
    return {"n_live": int(len(live)), "share_label_explained": explained / max(len(live), 1),
            "mean_best_auroc": float(np.mean([v["best_auroc"] for v in bins])) if bins else float("nan"),
            "mean_best_ap_over_base": float(np.mean([v["best_ap"] / max(v["base"], 1e-9) for v in bins])) if bins else float("nan"),
            "per_label": per_label, "relations": rel}


def liveness(idx: np.ndarray, val: np.ndarray, m: int, dense_at: float = 0.1) -> dict:
    n = len(idx)
    f = np.bincount(idx[val > 0].reshape(-1).astype(np.int64), minlength=m) / n
    return {"dead": int((f == 0).sum()), "dense": int((f > dense_at).sum()), "freq": f,
            "share_rare (< 2e-4)": float(((f > 0) & (f < 2e-4)).mean())}


def scorecard(sae, x_eval: torch.Tensor, downstream, idx: np.ndarray, val: np.ndarray,
              labels: dict[str, np.ndarray], kinds: dict[str, str],
              mean: torch.Tensor | None = None) -> tuple[dict, np.ndarray, dict]:
    """Faithfulness to the policy, liveness and label alignment of one dictionary on evaluation rows (x_eval the layer's
    activations, idx / val their compact codes, labels on the same rows): (card, firing frequency [m], every live
    feature's label relations)."""
    from .sae import downstream_check
    card = {"policy": downstream_check(sae, downstream, x_eval, mean=mean)}
    lv = liveness(idx, val, sae.m)
    card["dead"], card["dense"] = lv["dead"], lv["dense"]
    al = label_alignment(idx, val, labels, kinds, sae.m)
    card["labels"] = {k: v for k, v in al.items() if k != "relations"}
    return card, lv["freq"], al["relations"]


@torch.no_grad()
def activation_match(a, b, x: torch.Tensor, features, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """For the listed features of dictionary a: the feature of b whose activation correlates best with each over the
    rows x [n, d] (layer units), and that correlation. Catches matches the decoder cosine misses (a feature split in
    two, or the same events read along another direction)."""
    feats = torch.as_tensor(list(features), dtype=torch.long)
    F_, m = len(feats), b.m
    sa = torch.zeros(F_, dtype=torch.float64)
    sa2 = torch.zeros(F_, dtype=torch.float64)
    sb = torch.zeros(m, dtype=torch.float64)
    sb2 = torch.zeros(m, dtype=torch.float64)
    sab = torch.zeros(F_, m, dtype=torch.float64)
    n = 0
    for i in range(0, len(x), chunk):
        xb = x[i:i + chunk].float()
        za = a.encode(xb)[:, feats].double()
        zb = b.encode(xb).double()
        sa += za.sum(0)
        sa2 += (za * za).sum(0)
        sb += zb.sum(0)
        sb2 += (zb * zb).sum(0)
        sab += za.T @ zb
        n += len(xb)
    cov = sab / n - torch.outer(sa / n, sb / n)
    sd = torch.outer((sa2 / n - (sa / n) ** 2).clamp_min(0).sqrt(), (sb2 / n - (sb / n) ** 2).clamp_min(0).sqrt())
    r = torch.where(sd > 0, cov / sd.clamp_min(1e-30), torch.zeros_like(cov))
    v, j = r.max(1)
    return v.numpy(), j.numpy()
