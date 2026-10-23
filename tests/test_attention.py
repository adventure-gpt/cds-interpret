"""Per-head attention maps: probabilities recomputed from each module's own projections match a direct computation
(nn.MultiheadAttention, transformer layers on their fast path, custom modules), the aggregation by token name, the
shifts by situation, the Study method, the report and the command line."""

import json
import math
import textwrap

import numpy as np
import pytest
import torch

import cds_interpret as ci
from cds_interpret import attention as AT
from cds_interpret.errors import CDSInterpretError, LayerNotFound, ShapeError


def direct(q: torch.Tensor, k: torch.Tensor, heads: int, keep: torch.Tensor | None = None) -> torch.Tensor:
    """softmax(Q K^T / sqrt(d)) per head, written out the plain way: [B, H, Q, K]."""
    b, nq, d = q.shape
    dh = d // heads
    out = torch.zeros(b, heads, nq, k.shape[1])
    for h in range(heads):
        qh, kh = q[..., h * dh:(h + 1) * dh], k[..., h * dh:(h + 1) * dh]
        s = qh @ kh.transpose(1, 2) / math.sqrt(dh)
        if keep is not None:
            s = s.masked_fill(~keep[:, None, :], float("-inf"))
        out[:, h] = torch.softmax(s, -1)
    return out


# ------------------------------------------------------------------------------------- the built-in modules
@pytest.mark.parametrize("batch_first", [True, False])
def test_multihead_attention_matches_its_own_weights(batch_first):
    torch.manual_seed(0)
    mha = torch.nn.MultiheadAttention(16, 4, batch_first=batch_first).eval()
    x = torch.randn(5, 3, 16)
    xin = x if batch_first else x.transpose(0, 1)
    pad = torch.tensor([[False, False, True]] * 5)
    _, want = mha(xin, xin, xin, key_padding_mask=pad, average_attn_weights=False)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.att = mha

        def forward(self, z):                        # (need_weights=False: the fused path, no weights handed out)
            return self.att(z, z, z, key_padding_mask=pad, need_weights=False)[0]

    net = Net()
    before = net(xin)
    p = AT.probs(net, xin)
    assert list(p) == ["att"] and p["att"].shape == (5, 4, 3, 3)
    assert torch.allclose(p["att"], want, atol=1e-6)
    assert torch.equal(net(xin), before)              # reading changes nothing
    assert float(p["att"][..., 2].abs().max()) == 0.0  # the padded key gets nothing


def test_multihead_attention_kdim_float_mask_and_unbatched():
    torch.manual_seed(1)
    mha = torch.nn.MultiheadAttention(16, 2, kdim=8, vdim=8, batch_first=True).eval()
    q, k = torch.randn(4, 3, 16), torch.randn(4, 5, 8)
    mask = torch.randn(3, 5)
    _, want = mha(q, k, k, attn_mask=mask, average_attn_weights=False)
    tap = AT.AttentionTap(mha, {"x": ""})
    with tap:
        mha(q, k, k, attn_mask=mask, need_weights=False)
    assert torch.allclose(tap.probs["x"], want, atol=1e-6)
    _, want1 = mha(q[0], k[0], k[0], attn_mask=mask, average_attn_weights=False)
    with tap:
        mha(q[0], k[0], k[0], attn_mask=mask)
    assert torch.allclose(tap.probs["x"][0], want1, atol=1e-6)


@pytest.mark.parametrize("norm_first", [True, False])
def test_transformer_layer_on_its_fast_path(norm_first):
    torch.manual_seed(2)
    layer = torch.nn.TransformerEncoderLayer(16, 4, 32, dropout=0.0, batch_first=True, norm_first=norm_first).eval()
    x = torch.randn(6, 4, 16)
    with torch.no_grad():
        p = AT.probs(layer, x)
        h = layer.norm1(x) if norm_first else x
        _, want = layer.self_attn(h, h, h, average_attn_weights=False)
    assert torch.allclose(p[""], want, atol=1e-6)
    assert list(AT.find_sites(layer)) == [""]        # the layer is read, not its self_attn


@pytest.mark.filterwarnings("ignore:The PyTorch API of nested tensors")
def test_transformer_stack_with_padding_nested_tensors():
    torch.manual_seed(3)
    enc = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(16, 4, 32, dropout=0.0, batch_first=True), 2,
                                      enable_nested_tensor=True).eval()
    x = torch.randn(6, 4, 16)
    pad = torch.zeros(6, 4, dtype=torch.bool)
    pad[:3, 3] = True
    with torch.no_grad():
        p = AT.probs(enc, x, forward=lambda z: enc(z, src_key_padding_mask=pad))
        _, want = enc.layers[0].self_attn(x, x, x, key_padding_mask=pad, average_attn_weights=False)
    assert list(p) == ["layers.0", "layers.1"]
    assert torch.allclose(p["layers.0"][3:], want[3:], atol=1e-5)
    assert float(p["layers.0"][:3, :, :3, 3].abs().max()) == 0.0
    assert torch.allclose(p["layers.0"][:3, :, :3], want[:3, :, :3], atol=1e-5)


# ----------------------------------------------------------------------------------------- your own modules
class Block(torch.nn.Module):
    """A pre-norm entity-attention block with one packed q, k, v Linear and a [B, T] valid mask (True = attend),
    run through the fused kernel (scaled_dot_product_attention), which never hands its weights out."""

    def __init__(self, d: int = 16, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.ln1 = torch.nn.LayerNorm(d)
        self.qkv = torch.nn.Linear(d, 3 * d)
        self.proj = torch.nn.Linear(d, d)

    def forward(self, x, valid):
        n, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(n, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :])
        return x + self.proj(o.transpose(1, 2).reshape(n, t, d))


class EntityNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Linear(5, 16)
        self.blocks = torch.nn.ModuleList([Block(), Block()])
        self.head = torch.nn.Linear(16 * 4, 6)

    def forward(self, x):                            # x [N, 4 * 6]: 4 tokens of 5 features and a "present" flag
        tok = x.view(len(x), 4, 6)
        valid = tok[..., 5] > 0.5
        t = self.embed(tok[..., :5])
        for b in self.blocks:
            t = b(t, valid)
        return self.head(t.flatten(1))


def entity_inputs(n: int = 32, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, 6, generator=g)
    x[..., 5] = 1.0
    x[: n // 2, 3, 5] = 0.0                          # the 4th token absent in half the moments
    return x.reshape(n, 24)


def test_recompute_packed_qkv_matches_direct_and_rebuilds_the_output():
    torch.manual_seed(4)
    net = EntityNet().eval()
    x = entity_inputs()
    sites = {f"L{i + 1}": AT.Site(f"blocks.{i}", AT.recompute(qkv="qkv", heads="heads", norm="ln1", keep="valid"))
             for i in range(2)}
    ins = {}
    hook = net.blocks[0].register_forward_pre_hook(lambda m, a: ins.setdefault("x", a))
    p = AT.probs(net, x, sites)
    hook.remove()
    xb, valid = ins["x"]
    blk = net.blocks[0]
    h = blk.ln1(xb)
    qkv = blk.qkv(h)
    want = direct(qkv[..., :16], qkv[..., 16:32], 4, keep=valid)
    assert torch.allclose(p["L1"], want, atol=1e-6)
    v = qkv[..., 32:].view(len(x), 4, 4, 4).transpose(1, 2)              # [B, H, T, d]
    rebuilt = xb + blk.proj((p["L1"] @ v).transpose(1, 2).reshape(len(x), 4, 16))
    with torch.no_grad():
        assert torch.allclose(rebuilt, blk(xb, valid), atol=1e-5)
    assert float(p["L2"][: len(x) // 2, ..., 3].abs().max()) == 0.0      # the absent token


def test_separate_projections_masked_argument_and_the_protocol():
    torch.manual_seed(5)

    class Cross(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.n_heads = 2
            self.q_proj, self.k_proj = torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)

        def forward(self, x, padding=None):
            return x

    m = Cross()
    x = torch.randn(3, 5, 8)
    pad = torch.zeros(3, 5, dtype=torch.bool)
    pad[0, 4] = True
    fn = AT.recompute(q="q_proj", k="k_proj", masked="padding")
    tap = AT.AttentionTap(m, {"c": AT.Site("", fn)})
    with tap:
        m(x, padding=pad)
    want = direct(m.q_proj(x), m.k_proj(x), 2, keep=~pad)
    assert torch.allclose(tap.probs["c"], want, atol=1e-6)

    class WithMethod(Block):
        def attention_probs(self, x, valid):
            h = self.qkv(self.ln1(x))
            return AT.softmax_attention(h[..., :16], h[..., 16:32], self.heads, keep=valid)

    net = EntityNet()
    net.blocks[1] = WithMethod()
    assert list(AT.find_sites(net)) == ["blocks.1"]
    p = AT.probs(net.eval(), entity_inputs())
    assert p["blocks.1"].shape == (32, 4, 4, 4)

    AT.register_attention(Block, AT.recompute(qkv="qkv", norm="ln1", keep="valid"))
    try:
        assert list(AT.find_sites(EntityNet())) == ["blocks.0", "blocks.1"]
    finally:
        AT._REGISTRY.pop(Block)


def test_friendly_errors():
    net = EntityNet()
    with pytest.raises(CDSInterpretError, match="attention_probs"):
        AT.AttentionTap(net, {"x": "blocks.0"})                    # a custom module with no way to read it
    with pytest.raises(CDSInterpretError, match="No attention modules"):
        AT.as_sites(ci.demo.policy().net, None)
    with pytest.raises(CDSInterpretError, match="either qkv"):
        AT.recompute(q="q_proj")
    critic = torch.nn.TransformerEncoderLayer(8, 2, 16, batch_first=True)

    class AC(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = torch.nn.TransformerEncoderLayer(8, 2, 16, batch_first=True)
            self.critic = critic

    ac = AC().eval()
    with pytest.raises(LayerNotFound, match="never run"):
        list(AT.iter_probs(ac, ac.actor, torch.randn(4, 3, 8), {"c": "critic"}))
    with pytest.raises(ShapeError, match="token names"):
        AT.measure(ac, ac.actor, torch.randn(4, 3, 8), tokens=["a", "b"])


# ------------------------------------------------------------------------------------------------ aggregation
def test_aggregate_by_role_absent_tokens_entropy_and_self():
    p = torch.zeros(1, 1, 4, 4)
    p[0, 0] = torch.tensor([[0.5, 0.2, 0.3, 0.0],      # own car
                            [0.1, 0.6, 0.3, 0.0],      # ball
                            [0.2, 0.2, 0.6, 0.0],      # opponent A
                            [0.25, 0.25, 0.25, 0.25]])  # opponent B: absent at this moment
    roles = torch.tensor([[0, 1, 2, -1]])
    ag = AT.aggregate(p, roles, 3)
    assert torch.allclose(ag["full"][0, 0], torch.tensor([[0.5, 0.2, 0.3], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6]]))
    assert torch.allclose(ag["focus"][0, 0], torch.tensor([0.8, 1.0, 1.2]) / 3)
    assert ag["present"].tolist() == [[True, True, True]]
    assert ag["self"][0, 0].item() == pytest.approx((0.5 + 0.6 + 0.6) / 3)
    ent = [-(r * np.log(r)).sum() for r in ([0.5, 0.2, 0.3], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6])]
    assert ag["entropy"][0, 0].item() == pytest.approx(np.mean(ent), rel=1e-5)
    assert ag["spread"][0, 0].item() == pytest.approx(np.mean(ent) / np.log(3), rel=1e-5)
    two = AT.aggregate(p, torch.tensor([[0, 1, 2, 2]]), 3)          # both opponents: one role, summed as keys
    assert two["full"][0, 0, 0].tolist() == pytest.approx([0.5, 0.2, 0.3])
    assert two["full"][0, 0, 2].tolist() == pytest.approx([0.225, 0.225, 0.55])   # their rows averaged


def test_shift_stats_finds_a_planted_shift_and_not_a_fake_one():
    rng = np.random.default_rng(0)
    n, games = 6000, np.repeat(np.arange(40), 150)
    on = rng.random(n) < 0.3
    focus = np.zeros((n, 1, 1, 2))
    focus[..., 0, 0, 0] = 0.6 + 0.2 * on + rng.normal(0, 0.05, n)
    focus[..., 0, 0, 1] = 1 - focus[..., 0, 0, 0]
    fake = rng.random(n) < 0.3
    sits = {"real": AT.Situation(on, np.ones(n, bool), "when it is real"),
            "fake": AT.Situation(fake, np.ones(n, bool), "when it is fake"),
            "rare": AT.Situation(np.arange(n) < 10, np.ones(n, bool), "rarely")}
    st = AT.shift_stats(focus, games, sits, boot=200)
    r = st["real"]
    assert r["shift"][0, 0, 0] == pytest.approx(0.2, abs=0.01) and r["lo"][0, 0, 0] > 0.15
    assert r["tv"][0, 0] == pytest.approx(0.2, abs=0.01)
    f = st["fake"]
    assert f["lo"][0, 0, 0] < 0 < f["hi"][0, 0, 0]
    assert "skipped" in st["rare"]


# ------------------------------------------------------------------------------------------- the whole study
def test_study_attention_on_the_demo_bot(tmp_path):
    policy = ci.demo.attention_policy()
    moments = ci.demo.synthetic_moments(n_games=24, per_game=40)
    study = ci.Study(policy, moments, layers=[], log=None)
    maps = study.attention(tokens=ci.demo.TOKENS)
    assert maps.layers == ["L1", "L2"] and maps.paths == {"L1": "layers.0", "L2": "layers.1"}
    assert maps.head_names == ["L1 H1", "L1 H2", "L2 H1", "L2 H2"]
    assert maps.keys == ["own car", "ball", "opponent"]
    assert np.allclose(maps.focus.sum(-1), 1.0, atol=1e-5)
    assert np.allclose(np.nansum(maps.mean(), -1), 1.0, atol=1e-5)
    desc = maps.describe()
    assert set(desc) == set(maps.head_names) and all(t.startswith(h) for h, t in desc.items())
    assert "kickoff" in maps.shifts and "ball in the air" in maps.shifts
    text = maps.summary()
    assert "L2 H2" in text and "In words" in text
    # the same maps straight from the probabilities, by hand
    with torch.no_grad():
        p = AT.probs(policy.net, study.x)
    assert np.allclose(maps.focus[:, 0], p["layers.0"].mean(2).numpy(), atol=1e-5)
    page = study.report(tmp_path / "r.html").read_text(encoding="utf-8")
    assert "What each attention head looks at" in page and "who looks at whom" in page
    js = json.loads(maps.save(tmp_path / "a.json").read_text(encoding="utf-8"))
    assert js["keys"] == ["own car", "ball", "opponent"] and len(js["per_head"]) == 4
    assert set(js["per_head"][0]["map"]) == {"own car", "ball", "opponent"}
    figs = AT.figures(maps)
    assert {"focus", "heads", "shifts"} <= set(figs) and all(v.startswith("<svg") for v in figs.values())
    # one query token's row, roles and named situations
    mine = study.attention(tokens=ci.demo.TOKENS, query="own car", situations=["kickoff", "ball in the air"])
    assert np.allclose(mine.focus[:, 0], p["layers.0"][:, :, 0].numpy(), atol=1e-5)
    assert set(mine.shifts) == {"kickoff", "ball in the air"}
    grouped = study.attention(tokens=AT.Tokens(["me", "the rest"], roles=lambda x: np.tile([0, 1, 1], (len(x), 1))))
    assert np.allclose(grouped.focus[..., 1], maps.focus[..., 1] + maps.focus[..., 2], atol=1e-5)
    with pytest.raises(CDSInterpretError, match="No situation"):
        study.attention(tokens=ci.demo.TOKENS, situations=["raining"])
    with pytest.raises(CDSInterpretError, match="query"):
        study.attention(tokens=ci.demo.TOKENS, query="goalpost")


# ---------------------------------------------------------------------------------------------- command line
def test_cli_attention_demo(tmp_path, capsys):
    from cds_interpret.cli import main
    assert main(["attention", "--demo", "--source", "synthetic", "--out", str(tmp_path / "a")]) == 0
    out = capsys.readouterr().out
    assert "What each attention head looks at" in out and "In words" in out
    for f in ("attention.html", "attention.json", "focus.svg", "heads.svg", "shifts.svg"):
        assert (tmp_path / "a" / f).stat().st_size > 0


def test_cli_attention_on_a_recording(tmp_path, monkeypatch, capsys):
    (tmp_path / "tinybot.py").write_text(textwrap.dedent("""
        import torch
        import cds_interpret as ci

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                torch.manual_seed(0)
                self.embed = torch.nn.Linear(10, 16)
                self.att = torch.nn.MultiheadAttention(16, 2, batch_first=True)
                self.out = torch.nn.Linear(48, 90)

            def forward(self, x):
                t = self.embed(x.view(len(x), 3, 10))
                t = t + self.att(t, t, t, need_weights=False)[0]
                return self.out(t.flatten(1))

        def policy():
            return ci.Policy(Net().eval(), obs=ci.demo.obs, actions=ci.lookup_table(), name="tiny")
    """), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    rec = tmp_path / "rec"
    m = ci.demo.synthetic_moments(n_games=12, per_game=30)
    m.obs = ci.demo.obs(m.frames, m.me)
    m.save(rec)
    from cds_interpret.cli import main
    args = ["attention", str(rec), "--policy", "tinybot:policy", "--tokens", "a,b,c", "--out", str(tmp_path / "o"),
            "--max-moments", "200"]
    assert main(args) == 0
    js = json.loads((tmp_path / "o" / "attention.json").read_text(encoding="utf-8"))
    assert js["moments"] == 200 and js["keys"] == ["a", "b", "c"] and js["paths"] == {"L1": "att"}
    assert main(["attention", str(rec)]) == 1                     # no policy: a friendly message
    assert "needs a recording and your policy" in capsys.readouterr().out


def test_layers_with_different_numbers_of_heads():
    torch.manual_seed(6)

    class Two(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.MultiheadAttention(8, 2, batch_first=True)
            self.b = torch.nn.MultiheadAttention(8, 4, batch_first=True)

        def forward(self, x):
            t = x.view(len(x), 3, 8)
            t = t + self.a(t, t, t)[0]
            return t + self.b(t, t, t)[0]

    net = Two().eval()
    x = torch.randn(400, 24)
    maps = AT.measure(net, net, x, tokens=["p", "q", "r"])
    assert maps.heads == {"L1": 2, "L2": 4} and maps.head_names[:3] == ["L1 H1", "L1 H2", "L2 H1"]
    assert np.isnan(maps.focus[:, 0, 2:]).all() and np.isfinite(maps.focus[:, 1]).all()
    on = (x[:, 0] > 0).numpy()
    st = AT.shift_stats(maps.focus, np.repeat(np.arange(20), 20), {"s": AT.Situation(on, np.ones(400, bool), "w")})
    assert "shift" in st["s"] and len(maps.describe()) == 6
