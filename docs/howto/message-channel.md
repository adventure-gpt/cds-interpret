# Decode a team-mate message channel

<p align="center">
  <img src="../../assets/message-channel.svg" alt="Car A decides an action and a symbol; its team-mate B reads the symbol at its next decision. Three questions: usage, meaning, listening" width="560">
</p>

Some team bots learn to **talk**. Every decision, each car's network picks one of a few symbols as well as its action,
and its team-mates read that symbol as an input at their next decision. Nobody tells the symbols what to mean. Training
either finds a use for them or it doesn't.

This tool reads such a channel and answers three questions:

1. **Usage.** Which symbols get sent, and how sure is each decision about its symbol?
2. **Meaning.** What do the symbols say about the game? For each symbol you get a sentence such as *"sent mostly when
   the ball is in its own half (1.9x as often), especially if it is its team's closest car to the ball too (2.0x)"*.
3. **Listening.** Does a team-mate play differently when it gets a different symbol, or none? This is the question
   that decides. A symbol that tracks the game perfectly is still not a message if nobody's play depends on it.

## Try it in five minutes

The toolkit ships a toy 2v2 team that talks. Each car sends one of 4 symbols that says where it stands:

| symbol | meaning |
|---|---|
| 0 | closest to the ball, with the ball in their half |
| 1 | closest to the ball, with the ball in our half |
| 2 | a team-mate is closer to the ball, and I'm last back |
| 3 | a team-mate is closer to the ball, and I'm up-field |

The team-mate's network plays differently for each symbol. You can also make a deaf version of the team, which gets
the symbols and ignores them.

```python
import cds_interpret as ci

team = ci.demo.message_team()                       # or message_team(listens=False) for the deaf one
study = ci.Study(team.policy, team.moments, layers=[])
report = study.messages(logits=team.logits, channel=team.channel)
print(report.summary())
study.report("messages.html")                       # the page, with a "Message channel" section
```

It runs in a few seconds and prints this:

```text
Message channel: 4 symbols, 6,000 decisions
  USAGE: 4 of 4 symbols are sent (1% of decisions or more); a decision's message distribution holds 0.90 of 2.00
    possible bits of uncertainty (its top symbol 79% on average), and the sent symbols together 2.00 bits.
  MEANING: Yes - the symbols as sent carry information about the situation: 12 of 28 labels beat every time-shifted
    copy by 0.005 bits or more; the most about closest (0.807 bits, 81% of it), ball_half (0.478 bits, 48% of it), ...
  LISTENING: Yes - team-mates listen: swapping the symbol a receiver got for another changes its chosen action in
    86.13% of decisions (KL 2.6339 bits), silencing it in 73.85% (KL 1.1746 bits); ...
  symbols as sent (share, then mean probability):
    0:  25.3%   25.2%  sent mostly when the ball is in the opponents' half (1.9x as often), especially if it is its
                       team's closest car to the ball too (2.0x)
    1:  25.1%   25.2%  sent mostly when the ball is in its own half (1.9x as often), especially if it is its team's
                       closest car to the ball too (2.0x)
    2:  25.2%   25.2%  sent mostly when a team-mate is closer to the ball (2.0x as often), especially if it is the
                       last car back too (1.9x)
    3:  24.4%   24.4%  sent mostly when a team-mate is further back (2.0x as often), especially if a team-mate is
                       closer to the ball too (2.0x)
```

The tool was never told the rule, yet it found every part of it. Now make the deaf team (`listens=False`). Usage and
meaning come out exactly the same, but the listening answer changes:

```text
  LISTENING: No - team-mates don't listen: swapping the symbol a receiver got for another changes its chosen action
    in 0.00% of decisions (KL 0.0000 bits), silencing it in 0.00% ...
```

That gap between the two teams is the reason the third question exists.

## On your own bot

You need three things:

- **Your recorded moments.** Record team games ([Record games](record-moments.md)) and make a `Study` as usual.
  `layers=[]` is fine here, since this tool doesn't use probes.
- **The message logits.** These are what the message head gave at each moment, either as an array of shape
  `[moments, K]` or as a function `fn(inputs) -> [N, K]` that the toolkit runs on the study's inputs.
- **Where the symbols sit in the input.** You describe this with a `Channel`: for each team-mate slot, the K columns
  that hold that team-mate's symbol as a one-hot (one column per symbol, in order).

```python
from cds_interpret import messages as M

channel = M.Channel(symbols=8, columns=[range(600, 608), range(650, 658)])   # two team-mate slots
report = study.messages(logits=lambda x: net.message_head(x), channel=channel)
```

A slot counts as **live** at a moment when its columns hold a one-hot. A silent slot, an opponent's slot or an empty
slot is all zeros and is skipped. If your input marks team-mates some other way, pass
`live=lambda x: x[:, mate_columns] > 0.5`.

The symbols are taken to be the logits' argmax, as a bot that plays its most likely choice sends them. Pass `sent=` if
yours sends something else (sampled symbols, say). Without a `channel`, you get usage and meaning only.

Everything is also available without a `Study`:

```python
report = M.analyze(logits, labels=my_labels, games=game_ids, order=steps, x=inputs,
                   forward=policy.log_probs, channel=channel)
```

## How each answer is worked out

**Usage.**
- Each decision's message distribution (the softmax of its logits) has an entropy: from 0 bits for a sure decision to
  log2 K for an even one (3 bits for 8 symbols).
- The symbols sent across all decisions have an entropy too. Because an argmax symbol is decided by the moment, that
  number is also how much the symbol can say about the moment at most.
- `info_sampled` is the same for a symbol drawn at random from each decision's distribution, the way a bot in
  training sends it. It is the entropy of the average distribution minus the average entropy.

**Meaning.**
- For each label, the tool measures the **mutual information** between the symbol and the label, in bits. Numbers are
  cut into quarters first.
- Mutual information is never below zero, and a finite recording always shows a little by chance. So the label is also
  **shifted in time** inside each car's game, 20 times, and measured again. Each shifted copy keeps the label's own
  rhythm (how long a kickoff or a defence lasts) but puts it out of step with the symbols.
- A label counts when the real number beats every shifted copy by 0.005 bits or more. `excess` is how far above the
  copies' average it is, and `share` is that excess as a part of the label's own entropy.
- The **phrase** for each symbol comes from lifts: how much more often a situation holds when that symbol is sent
  than overall. The first part is the strongest such situation. The second part is the one that adds most *among
  those moments*, so a label that only repeats the first (where the ball lands, after "the ball is in their half") is
  passed over.

**Listening.**
- For every moment and every live team-mate slot, the receiver's action distribution is computed several ways: with
  the symbol it really got, with each of the K symbols in its place, and with silence.
- The tool reports the difference from the real one as **KL** (in bits) and as how often the **chosen action
  changes**.
- Every version of a moment goes through the network in the same batch, so rounding can't fake a change: putting the
  real symbol back gives exactly 0 (`noise_max` in the numbers).
- `references` are other edits of the same slot to compare against, such as a team-mate's boost changed. They tell you
  how big "a change" is for this network.
- The verdict thresholds:
  - **listens:** the KL is 0.02 bits or more, or the chosen action changes in 5% of decisions or more;
  - **barely:** 0.002 bits or more, or 0.5% of decisions or more;
  - **doesn't listen:** anything less.

## Reading it well

- **Meaning without listening is not communication.** A channel can track the game closely only because the
  network's message head reads the same features as its action head, while no team-mate does anything with it. The
  deaf demo team shows exactly that.
- **Argmax symbols can carry information that sampled ones don't.** If each decision's distribution is nearly even
  (entropy close to log2 K), training's sampled symbols are nearly random. The argmax of those almost-even logits can
  still follow the game closely. Look at `info_sampled` and the sampled column of the label table next to the as-sent
  numbers before you say what the bot "says".
- **A small KL can still flip actions.** A policy whose best actions are near-ties changes its argmax under tiny
  nudges. Compare the message's numbers with a reference edit of the same size and kind before you call it listening.
- **Moments that follow each other are alike.** That's why the baseline shifts labels in time instead of shuffling
  moments. Record whole games, and many of them.
- **Random moments are for trying the tool,** not for findings. The demo team's moments are independent random states.

## The pieces

`cds_interpret.messages`:
- `Channel` (`incoming`, `put`);
- `usage`, `meaning`, `listen`, `listening`, `verdicts`, `analyze`;
- `MessageReport` (`summary`, `phrases`, `label_table`, `figures`, `to_dict`, `save`);
- `situations` (the extra labels `closest`, `ball_half` and `mate_dist`);
- the helpers `entropy_bits`, `discretize`, `mutual_info`, `mutual_info_soft`, `lifts`, `phrase` and `PHRASES`.

`Study.messages` keeps its result as `message_results`, which the study page shows. See the
[API reference](../api.md#message-channel).
