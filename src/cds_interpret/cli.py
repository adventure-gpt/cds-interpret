"""The command line: ``cds-interpret demo`` (the whole tour on the demo bot), ``cds-interpret doctor`` (what is
installed and ready), ``cds-interpret attention`` (what each attention head looks at) and the panel tools."""

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


def attention_cmd(args: Any, log: Any = print) -> int:
    """``cds-interpret attention``: a policy's attention heads on a recording - the summary printed, and in the
    output folder the page (attention.html), the numbers (attention.json) and every figure (.svg)."""
    from pathlib import Path

    import numpy as np

    from . import _specs, demo
    from . import attention as AT
    from .errors import CDSInterpretError
    from .moments import Moments
    from .study import Study
    from .viz import plots
    if args.demo:
        policy = demo.attention_policy()
        tokens: Any = list(demo.TOKENS) if args.tokens is None else None
        log("The attention demo bot copies a simple scripted player through two small transformer layers over three "
            "tokens: its own car, the ball and the opponent. Let's see what each head looks at.")
        log("")
        moments = demo.demo_moments(policy, args.source, args.games, args.seconds, log=log)
    else:
        if not args.policy or not args.moments:
            raise CDSInterpretError("cds-interpret attention needs a recording and your policy: cds-interpret "
                                    "attention recordings/my-bot --policy my_bot:policy --tokens \"own car,ball,"
                                    "opponent\" (or try it on the demo bot: cds-interpret attention --demo)")
        policy = _specs.build(args.policy, "policy")
        root = Path(args.moments)
        moments = Moments.load(root) if (root / "manifest.json").exists() else Moments.load_all(root)
        tokens = None
    if args.tokens is not None:
        spec = args.tokens.strip()
        tokens = _specs.build(spec, "tokens") if ":" in spec and "," not in spec else spec
    if args.max_moments and len(moments) > args.max_moments:
        rows = np.sort(np.random.default_rng(0).choice(len(moments), args.max_moments, replace=False))
        moments = moments.take(rows)
    sites = _specs.build(args.sites, "sites") if args.sites else None
    forward = None
    if args.critic:
        if policy.value_fn is None:
            raise CDSInterpretError("--critic needs a policy with a critic: ci.Policy(..., value=my_critic)")
        forward = policy.value_fn
    study = Study(policy, moments, layers=[], log=lambda m: log("  " + m.strip()))
    maps = study.attention(sites=sites, tokens=tokens, query=args.query, forward=forward)
    log("")
    log(maps.summary())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, svg in AT.figures(maps).items():
        plots.save(svg, out / f"{name.replace('shift: ', 'shift_').replace(' ', '_')}.svg")
    maps.save(out / "attention.json")
    page = study.report(out / "attention.html", title=f"The attention heads of {policy.name}")
    log("")
    log(f"The page with every map and shift is {page}; the numbers are in {out / 'attention.json'}.")
    return 0


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
    pn.add_argument("--compact", action="store_true", help="the compact panel (top-right, smaller) instead of the default")
    ck = sub.add_parser("check-panel", help="check a panel config, and that what it shows is in your pack")
    ck.add_argument("path", help="the panel config (a .toml file)")
    ck.add_argument("pack", nargs="?", help="the thoughts pack it will show (a .npz file)")
    ck.add_argument("--policy", help="\"package.module:factory\" as you give it to with_thoughts(policy=...): also try "
                                     "'why this move' and 'looking at' with it, and time them")
    wt = sub.add_parser("watch-template", help="write a display-only RLBot script that shows the thoughts of a bot "
                                               "you can't modify")
    wt.add_argument("folder", nargs="?", default="watch", help="where to write it (default: watch)")
    wt.add_argument("--force", action="store_true", help="overwrite files that exist")
    rf = sub.add_parser("refit-pack", help="carry a thoughts pack over to a newer checkpoint of the same bot")
    rf.add_argument("pack", help="the old pack (a .npz file)")
    rf.add_argument("--policy", required=True, help="the NEW checkpoint: \"package.module:factory\" returning its "
                                                     "cds_interpret.Policy")
    rf.add_argument("--old-policy", required=True, help="the checkpoint the old pack was made for, the same way")
    rf.add_argument("--out", required=True, help="a folder for the new pack and every step's result")
    rf.add_argument("--moments", help="a recording to use (a folder) instead of recording games in RocketSim")
    rf.add_argument("--games", type=int, default=24, help="games to record in RocketSim (default 24)")
    rf.add_argument("--seconds", type=float, default=90.0, help="seconds per game (default 90)")
    rf.add_argument("--team-size", type=int, default=1, help="cars per team in the recorded games (default 1)")
    rf.add_argument("--opponent", default="chaser", help="who it plays: self, chaser or idle (default chaser)")
    rf.add_argument("--min-r", type=float, default=0.5, help="how closely a new feature must fire with an old one "
                                                             "to take its name (default 0.5)")
    rf.add_argument("--threads", type=int, default=2, help="CPU threads for torch (default 2)")
    at = sub.add_parser("attention", help="what each attention head of a network looks at, by token and situation")
    at.add_argument("moments", nargs="?", help="a recording of your bot (a folder; one with several recordings works too)")
    at.add_argument("--policy", help="\"package.module:factory\" returning your cds_interpret.Policy")
    at.add_argument("--demo", action="store_true", help="the attention demo bot, on games it plays (or random moments)")
    at.add_argument("--tokens", help="the tokens' names, in order (\"own car,ball,opponent\"), or "
                                     "\"package.module:factory\" returning them (a list, or attention.Tokens)")
    at.add_argument("--sites", help="\"package.module:factory\" returning the attention layers to read ({name: path or "
                                    "attention.Site}); default: every attention module the network runs")
    at.add_argument("--query", help="read each head's row for this query token only (default: every token's, averaged)")
    at.add_argument("--critic", action="store_true", help="the critic's attention: run the policy's value function")
    at.add_argument("--out", default="cds-attention", help="the folder for the page, figures and numbers "
                                                           "(default: cds-attention)")
    at.add_argument("--max-moments", type=int, default=0, help="study at most this many moments, drawn at random "
                                                               "(default: all)")
    at.add_argument("--source", choices=("auto", "rocketsim", "synthetic"), default="auto",
                    help="--demo: where its moments come from (auto: RocketSim when it is ready, else synthetic)")
    at.add_argument("--games", type=int, default=12, help="--demo: games to record in RocketSim (default 12)")
    at.add_argument("--seconds", type=float, default=60.0, help="--demo: seconds per game (default 60)")
    at.add_argument("--threads", type=int, default=2, help="CPU threads for torch (default 2)")
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
        from pathlib import Path
        out = Path(args.path)
        if out.exists() and not args.force:
            print(f"{out} already exists - pass --force to overwrite it")
            return 1
        from .overlay.panel import preset_text
        out.write_text(preset_text("compact" if args.compact else "default"), encoding="utf-8")
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
    if args.cmd == "watch-template":
        from .errors import CDSInterpretError
        from .overlay.watch import write_template
        try:
            written = write_template(args.folder, args.force)
        except CDSInterpretError as e:
            print(e)
            return 1
        print("Wrote " + " and ".join(written) + ". Edit the three lines marked EDIT in watch.py, then add "
              "watch.script.toml to your match as a script (docs/howto/watch-a-bot.md).")
        return 0
    if args.cmd == "refit-pack":
        import torch
        torch.set_num_threads(max(1, args.threads))
        from .errors import CDSInterpretError
        from .refit import refit_pack
        try:
            refit_pack(args.pack, args.policy, args.old_policy, args.out, moments=args.moments, games=args.games,
                       seconds=args.seconds, team_size=args.team_size, opponent=args.opponent, min_r=args.min_r)
        except CDSInterpretError as e:
            print(e)
            return 1
        return 0
    if args.cmd == "attention":
        import torch
        torch.set_num_threads(max(1, args.threads))
        from .errors import CDSInterpretError
        try:
            return attention_cmd(args)
        except CDSInterpretError as e:
            print(e)
            return 1
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
