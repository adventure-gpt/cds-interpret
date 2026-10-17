# Coming from rlgym-ppo or GigaLearn

If you train bots with an rlgym-ppo-style Python trainer or a GigaLearn-style C++ trainer, you already have
everything this toolkit needs. This page maps the ideas you know onto it. (It describes concepts only; how your trainer
names or stores things may differ - check your own code.)

## The pieces you have, and where they go

| in your trainer | here |
|---|---|
| the **policy network** (the actor) | `ci.Policy(net, ...)` - just the actor; no optimizer, no critic, no trainer state |
| the **critic** (value network) | optional: `value=` on the Policy, or `critic=` for the live overlay's outlook bar |
| the **observation builder** | the `obs` function - or `ci.rlgym.obs_fn(builder)` for an RLGym v2 builder - or none at all if you record your bot in the real game or in your env |
| the **action parser** (a lookup table of controls) | `actions=` - the table itself, `[A, 8]` |
| the **action distribution** at the end of the actor | `output=`: `"logits"`, or `"probs"` if the network ends in a softmax |
| **tick skip** / action repeat | `tick_skip=` when recording in RocketSim (8 is usual) |
| **checkpoints** saved during training | snapshots for `ci.tracking.track`: when did a concept appear? |
| **self-play and opponent pools** | `opponent=` in `record.rocketsim`: itself, another checkpoint as a `Policy`, or a scripted player |
| **episode stats and rewards** | what you asked for. Probes and features tell you what it actually learned |

## A Python trainer (rlgym-ppo style)

The actor is usually a small MLP saved as a state dict. Rebuild the same module, load the weights, wrap it:

```python
import torch
import cds_interpret as ci

actor = MyActorClass(obs_size, layer_sizes, n_actions)       # the class your trainer uses
actor.load_state_dict(torch.load("checkpoints/latest/policy.pt", map_location="cpu"))
policy = ci.Policy(actor, obs=my_obs, actions=my_table, output="probs")   # "probs" if it ends in a softmax
```

If the actor's layers live in an `nn.Sequential` attribute and its `forward` does something extra (sampling, say), point
`forward=` at the part that returns the distribution: `forward=actor.layers` (an attribute that is a module) works.

The observation: if your trainer uses RLGym v2 obs builders, `ci.rlgym.obs_fn(MyObs())` runs yours on any moment. If it
uses the older (v1-style) state objects, either write a small `obs(frames, me)` in your builder's image, or skip the
question entirely: record the bot in your own env (`ci.record.rlgym`, for v2 envs) or in the real game
(`ci.record.with_recorder`), and the moments carry the exact inputs.

Shared-weight attention or per-car encoders work the same way; study a layer after the pooling, where one vector
describes the moment.

## A C++ trainer (GigaLearn style)

The toolkit lives in Python, so the network has to be loadable in PyTorch:

- **TorchScript file** (a `.pt` that `torch.jit.load` opens): `net = ci.eager_copy(torch.jit.load(path))` rebuilds MLP
  building blocks as hookable modules with the same weights. Custom modules: rebuild the class in Python and load
  `torch.jit.load(path).state_dict()`.
- **Weights in your own format**: write them out as arrays (one per layer), rebuild the MLP in Python (a few lines of
  `nn.Sequential`) and copy them in. Check the rebuilt network gives the same outputs as the C++ one on a few inputs.

The observation builder is C++ too. Port it to a Python `obs(frames, me)` - usually the most work, and worth checking
against the C++ outputs on recorded states - or record in the real game if your RLBot bot runs in Python. With neither,
the **watch** tier still works on any bot: labels, habits and drawings from RLBot packets or recordings, no network
needed.

## Using it while you train

- **Is it learning the right things?** Probe a checkpoint every so often for the concepts your reward is about
  (`threat`, `on_target`, `first_to_ball`, ...). A concept that never becomes readable is a sign the bot is winning
  some other way.
- **Reward hacking.** SAE features that fire on odd, repeated situations are worth a look: they show what the policy
  has built its play around.
- **Comparing runs.** The same moments, probed in two runs' networks, compare what they learned, not just how they
  score.
- **A live view.** The thoughts overlay on a checkpoint in RLBot shows its reading of the game while you watch it play.

None of this needs a GPU, and none of it changes your trainer. The toolkit works on the network object you give it:
`ci.Policy(net)` puts it in evaluation mode (`net.eval()`), reads it through hooks, and an edit lasts only inside its
`with` block. To keep your training network untouched, hand the toolkit a copy - load the checkpoint into a fresh
instance, or `copy.deepcopy(net)`.
