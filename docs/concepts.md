# How it works

This page explains the ideas behind the toolkit in plain words: what it looks at, what each number means, and how to
read the results without fooling yourself. You don't need it to get started - the [quickstart](quickstart.md) is
enough for that - but it makes everything else easier to trust.

## The big picture

A Rocket League bot like yours looks at the game, runs the numbers through its network, and picks an action. The
network is a stack of **layers**: each layer turns the numbers it gets into new numbers and passes them on. Somewhere
in those layers, the bot works out things like "the ball is coming at my goal" or "I'll get there first" - but nobody
wrote those down. They are spread across hundreds of numbers.

The toolkit finds them in four ways:

1. it **records moments** of your bot playing, and works out what was true in each one (the **labels**);
2. it **asks questions** of each layer (**probes**): can this layer tell whether the ball is about to go in?
3. it **finds the ideas** the layer uses on its own (**features**, from a sparse autoencoder);
4. it **switches ideas off** to see whether the bot actually relies on them (**edits**).

And it can **show** what it finds, live, while the bot plays.

## What you need of a bot: three levels

How much you can learn depends on how much of the bot you have. Every level works for any bot.

| level | what you have | what you can do |
|---|---|---|
| **Watch** | games you can watch - RLBot packets, RLGym states, recordings | work out the labels of every moment, count habits (demos taken, where it sits when it's last back), draw moments from above |
| **Open** | its PyTorch network and how it builds its input | everything above, plus probes, features, edits, reports and the live overlay |
| **Run** | it plays in RocketSim or your RLGym environment | everything above, plus recording at will, testing an edit over whole games, and following it through training |

## Moments and frames

A **frame** is everything the game itself knows at one instant: where the ball and every car are, how fast they move,
which way they face, how much boost each has, who's on the ground. It uses the game's own units (Unreal units, about a
centimetre each) and the usual sides: blue defends the goal at negative y, orange the one at positive y.

A **moment** is a frame seen from your bot's car, plus what its network took in and gave out right then, and which
game it came from. The toolkit always keeps moments from whole games together, so it can learn from some games and
test on others.

## Labels: facts about each moment

A **label** is a fact about a moment, worked out from the game state alone - never from your bot's own input. That way
every bot is measured on the same footing. There are 36 of them, for example:

- **yes / no facts**: "the ball is going into our goal within 3 seconds" (`threat`), "nobody gets to the ball before
  me" (`first_to_ball`), "I'm the last player back" (`last_back`);
- **numbers**: "seconds until the ball lands" (`land_t`), "how far away the ball is" (`ball_dist`);
- **what happened next** (in recorded games): "we score within 3 seconds" (`goal_soon`), "I touch the ball within half
  a second" (`touch_soon`).

Each one is worked out from your car's side of the field. [The full list](labels.md) - and you can add your own.

## Layers

A **layer** is one stage of the network. The toolkit names layers by their path in your network (`"layers.3"`,
`"actor.body.1"`) and reads them through "hooks": small listeners PyTorch lets you attach to a layer. A hook that only
listens doesn't change anything - your network's outputs stay exactly the same, to the last bit.

The most interesting layers are usually the hidden ones, right after their activation function. The toolkit picks
those by default; `policy.layers(x)` lists them all.

## Probes: asking the network a question

A **probe** is a small readout that asks the network one question. It takes a layer's numbers, multiplies each by a
weight, and adds them up - nothing more. If such a simple readout can say "the ball is going in", the layer clearly
holds that information.

**How good is the answer?** The probe learns its weights on some games and is scored on games it never saw. The score
depends on the kind of question:

- for yes / no questions, **AUROC**: the chance that a "yes" moment scores higher than a "no" moment. 0.5 is a coin
  flip; 1.0 is perfect.
- for number questions, **R²**: how much of the number's variation it explains. 0 is no better than always guessing
  the average; 1.0 is perfect; it can go below 0.

**Compared with what?** A high score alone says little. The toolkit always asks the same question of two baselines:

- **the raw input**: your bot's own observation. If your observation includes your boost, then "how much boost do I
  have?" is easy everywhere - no discovery.
- **a random network**: the same shape, but random weights. Random layers keep a surprising amount of information, so
  a trained layer has to do *better* than that to have learned something.

The labels a layer reads clearly better than both are what the network worked out for itself.

**Steady enough to show?** A probe can score well and still jump around on screen. The toolkit measures that too:
how much the readout moves when the layer's numbers wobble a little, and how much it moves from one moment to the next
compared with the real answer. [More on probes you show live](howto/probes.md#probes-you-show-live).

**One caution:** a probe shows information is *there*. It doesn't show the bot *uses* it. That's what edits are for.

## Features: the ideas the network uses

A **sparse autoencoder** is a tool that finds the separate ideas a network uses. It learns to rewrite each moment's
layer as a mix of a few "features" out of many - say, 16 active out of 512. Features often line up with things a
person can name, where single numbers in the layer rarely do.

**Is the dictionary any good?** Two checks, on games it never saw:

- how much of the layer it rebuilds (the toolkit prints "keeps 97.5% of the layer's variance");
- whether the bot still plays the same with the rebuilt layer in place of the real one (the share of moments its choice
  changes, and the **KL divergence**: how far its whole action distribution moves - 0 means not at all).

**What is a feature about?** The toolkit compares each feature with every label: how well it picks out a yes / no fact
(AUROC, and **lift**: how many times likelier the fact is when the feature fires), or how closely it follows a number
(**correlation**, from -1 to 1). From that it suggests a name. A suggestion is a guess until the feature's strongest
moments agree with it - [the naming guide](howto/name-features.md) walks through that.

## Edits: does the bot rely on it?

An **edit** changes what a layer computes, every time the network runs. You can remove a feature, turn it up, remove
a probe's direction, or replace the whole layer with its average.

Then you measure what changed. On recorded moments: how often the bot's chosen action changes, and how far its action
distribution moves. Over whole games: the goal share, or how often something happens, with an error bar.

If removing the "threat on goal" idea changes what the bot does when its goal is threatened - and removing a random
idea of the same size doesn't - the bot relies on it. [Causal tests](howto/causal-tests.md) shows how.

## The live overlay

The overlay is the same machinery, live. A small **pack** file says which features and probes to read from which
layer. While your bot plays, a hook copies that layer (without changing it), a separate low-priority process reads the
pack from it, and a panel in the corner of the screen shows the result. [The overlay guide](howto/overlay.md) has the
details, and [you can design the panel yourself](howto/customize-the-panel.md).

## Reading results well

- **Ten games or more.** Scores from a handful of games are rough, especially for rare events like demos.
- **Always look at the baselines.** A high score that the raw input matches is not a discovery.
- **A name is a guess until the moments agree.** Look at the strongest moments before you trust a feature's name.
- **Test claims in games.** An edit that moves the policy is a claim about the network: check it over enough games
  for the error bar to rule out "no effect".
- **Random moments are for practice.** The demo's synthetic moments are fine for trying the toolkit, not for findings.
- **Keep data out of your repository.** Recordings and trained dictionaries can get big.
