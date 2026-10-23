# CDS-interpret documentation

New here? Start with the **[quickstart](quickstart.md)**: five minutes on the built-in demo bot, then the same on yours.

## Guides

- [Bring your own bot](howto/bring-your-own-policy.md) - wrap any PyTorch network: actor-critics, TorchScript, RLGym
  observations
- [Record games](howto/record-moments.md) - in RocketSim, in the real game through RLBot, in your RLGym environment,
  or from data you have
- [Ask your bot questions](howto/probes.md) - probes, their baselines, and probes steady enough to show live
- [Find the ideas it uses](howto/sae-features.md) - sparse autoencoders and their features
- [Name your bot's features](howto/name-features.md) - from numbers to names you trust, step by step
- [Test what it relies on](howto/causal-tests.md) - switch ideas off and up, on moments and over whole games
- [See what a decision rests on](howto/what-decisions-rest-on.md) - the ball, its car, the opponents; and what each
  feature makes the bot do
- [What each attention head looks at](howto/attention-heads.md) - per-head attention maps by token name, how they
  shift with the situation, and reading your own attention modules (fused kernels too)
- [Check your tools](howto/check-your-tools.md) - dictionaries true to the bot's choices and found again, probes the
  bot really uses, gauges whose numbers mean what they say
- [Watch it think in RLBot](howto/overlay.md) - the live thoughts overlay
- [Why this move, and what it's looking at](howto/why-and-looking.md) - the panel's two rows about each decision
- [Team modes](howto/team-modes.md) - 2v2 and 3v3: a landing ring for every car, a panel that follows the camera
- [Thoughts for a bot you can't modify](howto/watch-a-bot.md) - a display-only script over a bot that runs on its own
- [Keep your panel when your bot trains on](howto/keep-your-panel.md) - a pack carried over to a newer checkpoint
- [Design your own panel](howto/customize-the-panel.md) - every option of the panel's file, and Python hooks
- [Reports and drawings](howto/figures.md) - the study page, its sections, the drawings, habit counts
- [See what your bot learns between checkpoints](howto/track-training.md) - what each checkpoint picked up, and what
  to practise next

## Background

- [How it works](concepts.md) - the ideas behind the toolkit, in plain words
- [Labels](labels.md) - every fact the toolkit works out about a moment, and adding your own
- [Coming from rlgym-ppo or GigaLearn](coming-from.md) - where the things you know fit

## Reference

- [API reference](api.md) - every function and class
- [When something goes wrong](troubleshooting.md) - error messages and what to do
- [Contributing](../CONTRIBUTING.md) and the [changelog](../CHANGELOG.md)
