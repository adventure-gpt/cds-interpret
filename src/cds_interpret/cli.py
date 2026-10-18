"""The command line: ``cds-interpret demo`` (the whole tour on the demo bot) and ``cds-interpret doctor`` (what is
installed and ready)."""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
from typing import Any


def _version(mod: str) -> str | None:
    try:
        m = importlib.import_module(mod)
    except Exception:                                 # (absent, or broken: either way not ready)
        return None
    v = getattr(m, "__version__", None)
    if v is None:
        try:
            from importlib.metadata import version
            v = version({"RocketSim": "rocketsim", "rlgym": "rlgym-rocket-league"}.get(mod, mod))
        except Exception:
            v = "installed"
    return str(v)


def doctor(log: Any = print) -> dict[str, Any]:
    """What is installed and ready, and what to install for the rest."""
    from . import __version__
    from .errors import MeshesNotFound
    out: dict[str, Any] = {"cds-interpret": __version__, "python": platform.python_version()}
    for mod in ("torch", "numpy", "RocketSim", "rlgym", "rlbot"):
        out[mod] = _version(mod)
    meshes = None
    if out["RocketSim"]:
        from .record.sim import find_meshes
        try:
            meshes = str(find_meshes())
        except MeshesNotFound:
            meshes = None
    out["meshes"] = meshes
    log(f"cds-interpret {__version__} on Python {out['python']} ({platform.system()})")
    log(f"  torch      {out['torch']}")
    log(f"  numpy      {out['numpy']}")
    tip = {"RocketSim": 'pip install "cds-interpret[sim]"  (recording games in RocketSim)',
           "rlgym": 'pip install "cds-interpret[rlgym]"  (RLGym v2 bridges; brings RocketSim and its meshes; '
                    'Python 3.11-3.12)',
           "rlbot": 'pip install "cds-interpret[rlbot]"  (the live overlay and recording in the real game)'}
    for mod in ("RocketSim", "rlgym", "rlbot"):
        log(f"  {mod:10s} {out[mod] or 'not installed - ' + tip[mod]}")
    if out["RocketSim"]:
        log(f"  meshes     {meshes or 'not found - see docs/howto/record-moments.md (or install the rlgym extra)'}")
    ready = ["probes, features, edits, reports (always)"]
    if out["RocketSim"] and meshes:
        ready.append("recording in RocketSim")
    if out["rlgym"]:
        ready.append("RLGym bridges")
    if out["rlbot"]:
        ready.append("the RLBot overlay and recorder")
    log("Ready: " + "; ".join(ready))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cds-interpret", description="Look inside any PyTorch Rocket League bot.")
    sub = ap.add_subparsers(dest="cmd")
    d = sub.add_parser("demo", help="the whole tour on the demo bot: probes, features, edits, an HTML report")
    d.add_argument("--out", default="cds-demo", help="the folder for the report (default: cds-demo)")
    d.add_argument("--source", choices=("auto", "rocketsim", "synthetic"), default="auto",
                   help="where the moments come from (auto: RocketSim when it is ready, else synthetic)")
    d.add_argument("--games", type=int, default=12, help="games to record in RocketSim (default 12)")
    d.add_argument("--seconds", type=float, default=60.0, help="seconds per game (default 60)")
    d.add_argument("--threads", type=int, default=2, help="CPU threads for torch (default 2)")
    sub.add_parser("doctor", help="what is installed and ready")
    pn = sub.add_parser("panel", help="write an example panel config (every option, commented) to customize")
    pn.add_argument("path", nargs="?", default="panel.toml", help="where to write it (default: panel.toml)")
    pn.add_argument("--force", action="store_true", help="overwrite the file if it exists")
    ck = sub.add_parser("check-panel", help="check a panel config, and that what it shows is in your pack")
    ck.add_argument("path", help="the panel config (a .toml file)")
    ck.add_argument("pack", nargs="?", help="the thoughts pack it will show (a .npz file)")
    ck.add_argument("--policy", help="\"package.module:factory\" as you give it to with_thoughts(policy=...): also try "
                                     "'why this move' and 'looking at' with it, and time them")
    ap.add_argument("--version", action="store_true", help="print the version")
    args = ap.parse_args(argv)
    if args.version:
        from . import __version__
        print(__version__)
        return 0
    if args.cmd == "doctor":
        doctor()
        return 0
    if args.cmd == "panel":
        from importlib.resources import files
        from pathlib import Path
        out = Path(args.path)
        if out.exists() and not args.force:
            print(f"{out} already exists - pass --force to overwrite it")
            return 1
        out.write_text(files("cds_interpret.overlay").joinpath("panel_example.toml").read_text(encoding="utf-8"),
                       encoding="utf-8")
        print(f"Wrote {out}. Change what you like, then pass it to the overlay: with_thoughts(..., panel=\"{out}\")")
        return 0
    if args.cmd == "check-panel":
        from .errors import CDSInterpretError
        from .overlay.panel import PanelConfig, check_panel
        try:
            conf = PanelConfig.load(args.path)
            problems = check_panel(conf, args.pack) if args.pack else []
            tried = None
            if args.policy:
                if not args.pack:
                    print("--policy needs the pack too: cds-interpret check-panel panel.toml thoughts.npz --policy ...")
                    return 1
                from .overlay.explain import try_readings
                tried = try_readings(args.pack, args.policy, donors=conf.wants["donors"])
                problems += tried["problems"]
        except CDSInterpretError as e:
            print(e)
            return 1
        for p in problems:
            print(f"- {p}")
        if tried is not None:
            for what, key in (("why this move", "why_ms"), ("looking at", "looking_ms")):
                if tried[key] is not None:
                    print(f"  {what}: works, {tried[key]:.1f} ms a reading on one thread")
        if problems:
            return 1
        print(f"{args.path} is fine" + (f", and everything it shows is in {args.pack}." if args.pack else
                                         ". (Name your pack too, to check the rows' probes and features are in it.)"))
        return 0
    if args.cmd == "demo":
        import torch
        torch.set_num_threads(max(1, args.threads))
        from .demo import run
        run(args.out, source=args.source, games=args.games, seconds=args.seconds)
        print("Open it in your browser to explore.")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
