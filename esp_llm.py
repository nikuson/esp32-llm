#!/usr/bin/env python3
"""
esp_llm.py — host-side toolkit for the "LLM on ESP32-S3" experiment.

  python esp_llm.py analyze  --model stories260K.bin --tok tok512.bin
  python esp_llm.py export   --model stories260K.bin --out model.espl
  python esp_llm.py flash    --port /dev/cu.usbmodem1101 [--build]
  python esp_llm.py bench    --port /dev/cu.usbmodem1101
  python esp_llm.py verify   --port /dev/cu.usbmodem1101
  python esp_llm.py gen      --port /dev/cu.usbmodem1101 --prompt "Once upon a time"

Requires: numpy, pyserial (and esptool for `flash`).
"""
import argparse, glob, json, math, os, struct, subprocess, sys, time
import numpy as np

F32 = np.float32
HERE = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────── llama2.c checkpoint ───────────────────────────────

def load_llama2c(path):
    raw = open(path, "rb").read()
    dim, hidden, L, nh, nkv, vocab, seq = struct.unpack("7i", raw[:28])
    shared = vocab > 0
    vocab = abs(vocab)
    hs = dim // nh
    kv = dim * nkv // nh
    a = np.frombuffer(raw, dtype=F32, offset=28)
    o = 0
    def take(*shape):
        nonlocal o
        n = int(np.prod(shape))
        t = a[o:o + n].reshape(shape)
        o += n
        return t.copy()
    m = dict(dim=dim, hidden=hidden, L=L, nh=nh, nkv=nkv, vocab=vocab, seq=seq, hs=hs, kv=kv)
    m["emb"] = take(vocab, dim)
    m["rms_att"] = take(L, dim)
    m["wq"] = take(L, dim, dim)
    m["wk"] = take(L, kv, dim)
    m["wv"] = take(L, kv, dim)
    m["wo"] = take(L, dim, dim)
    m["rms_ffn"] = take(L, dim)
    m["w1"] = take(L, hidden, dim)
    m["w2"] = take(L, dim, hidden)
    m["w3"] = take(L, hidden, dim)
    m["rms_final"] = take(dim)
    o += seq * hs // 2 * 2          # legacy freq_cis tables, recomputed below
    m["cls"] = m["emb"] if shared else take(vocab, dim)
    m["shared"] = shared
    return m

# ─────────────────────────────── tokenizer (run.c port) ───────────────────────────────

class Tokenizer:
    def __init__(self, path, vocab_size):
        raw = open(path, "rb").read()
        self.max_len = struct.unpack_from("i", raw, 0)[0]
        o = 4
        self.vocab, self.scores = [], []
        for _ in range(vocab_size):
            s, n = struct.unpack_from("fi", raw, o); o += 8
            self.vocab.append(raw[o:o + n]); o += n
            self.scores.append(s)
        self.lookup = {t: i for i, t in enumerate(self.vocab)}

    def decode(self, prev, tok):
        p = self.vocab[tok]
        if prev == 1 and p.startswith(b" "):
            p = p[1:]
        if p.startswith(b"<0x") and p.endswith(b">") and len(p) == 6:
            return bytes([int(p[3:5], 16)])
        return p

    def encode(self, text, bos=True):
        toks = [1] if bos else []
        if text:
            toks.append(self.lookup[b" "])
        for ch in text.encode("utf-8").decode("utf-8"):
            b = ch.encode("utf-8")
            if b in self.lookup:
                toks.append(self.lookup[b])
            else:
                toks += [x + 3 for x in b]
        while True:
            best, bi, bid = -1e10, -1, -1
            for i in range(len(toks) - 1):
                m = self.vocab[toks[i]] + self.vocab[toks[i + 1]]
                j = self.lookup.get(m)
                if j is not None and self.scores[j] > best:
                    best, bi, bid = self.scores[j], i, j
            if bi < 0:
                break
            toks[bi:bi + 2] = [bid]
        return toks

def find_tok(model_path, tok_path):
    if tok_path:
        return tok_path
    for c in [os.path.join(os.path.dirname(model_path), n) for n in ("tok512.bin", "tokenizer.bin")]:
        if os.path.exists(c):
            return c
    sys.exit("tokenizer not found, pass --tok")

# ─────────────────────────────── float reference (== run.c) ───────────────────────────────

def rope_table(m, seq):
    hs = m["hs"]
    t = np.zeros((seq, hs // 2, 2), F32)
    for i in range(hs // 2):
        freq = 1.0 / F32(10000.0) ** (F32(2 * i) / F32(hs))
        ang = np.arange(seq, dtype=F32) * F32(freq)
        t[:, i, 0] = np.cos(ang)
        t[:, i, 1] = np.sin(ang)
    return t

def apply_rope(v, nheads, hs, cs):
    v = v.reshape(nheads, hs // 2, 2)
    c, s = cs[:, 0], cs[:, 1]
    a, b = v[..., 0].copy(), v[..., 1].copy()
    v[..., 0] = a * c - b * s
    v[..., 1] = a * s + b * c
    return v.reshape(-1)

class FloatModel:
    def __init__(self, m):
        self.m = m
        self.rope = rope_table(m, m["seq"])
        self.reset()

    def reset(self):
        m = self.m
        self.kc = np.zeros((m["L"], m["seq"], m["kv"]), F32)
        self.vc = np.zeros_like(self.kc)

    def forward(self, tok, pos):
        m = self.m
        D, hs, nh, nkv = m["dim"], m["hs"], m["nh"], m["nkv"]
        rep = nh // nkv
        x = m["emb"][tok].copy()
        for l in range(m["L"]):
            xb = rmsnorm_np(x, m["rms_att"][l])
            q = m["wq"][l] @ xb
            k = m["wk"][l] @ xb
            v = m["wv"][l] @ xb
            q = apply_rope(q, nh, hs, self.rope[pos])
            k = apply_rope(k, nkv, hs, self.rope[pos])
            self.kc[l, pos], self.vc[l, pos] = k, v
            out = np.zeros(D, F32)
            for h in range(nh):
                kh = h // rep
                K = self.kc[l, :pos + 1, kh * hs:(kh + 1) * hs]
                V = self.vc[l, :pos + 1, kh * hs:(kh + 1) * hs]
                sc = K @ q[h * hs:(h + 1) * hs] / np.sqrt(F32(hs))
                sc = np.exp(sc - sc.max()); sc /= sc.sum()
                out[h * hs:(h + 1) * hs] = sc @ V
            x = x + m["wo"][l] @ out
            xb = rmsnorm_np(x, m["rms_ffn"][l])
            h1, h3 = m["w1"][l] @ xb, m["w3"][l] @ xb
            x = x + m["w2"][l] @ (h1 / (1 + np.exp(-h1)) * h3)
        return m["cls"] @ rmsnorm_np(x, m["rms_final"])

def rmsnorm_np(x, w):
    return (w * (x / np.sqrt(np.mean(x * x) + F32(1e-5)))).astype(F32)

# ─────────────────────────────── quantization ───────────────────────────────

def ceil16(n):
    return (n + 15) // 16 * 16

def round_away(v):
    """round half away from zero, float32 in, int32 out — same as the firmware."""
    v = v.astype(F32)
    return np.trunc(v + np.where(v >= 0, F32(0.5), F32(-0.5)).astype(F32)).astype(np.int32)

def quant_rows(w, bits=8, group=None):
    """symmetric per-row (or per-group) quantization -> (int8 values, float32 scales, dequant)."""
    qmax = 2 ** (bits - 1) - 1
    w = w.astype(F32)
    rows, cols0 = w.shape
    g = group or cols0
    if cols0 % g:
        w = np.concatenate([w, np.zeros((rows, g - cols0 % g), F32)], 1)
    cols = w.shape[1]
    wg = w.reshape(rows, cols // g, g)
    amax = np.abs(wg).max(-1)
    s = np.where(amax > 0, amax / F32(qmax), F32(1)).astype(F32)
    q = np.clip(round_away(wg * (F32(1) / s)[..., None]), -qmax, qmax).astype(np.int8)
    deq = (q.astype(F32) * s[..., None]).reshape(rows, cols)[:, :cols0]
    return q.reshape(rows, cols)[:, :cols0], (s if group else s[:, 0]), deq

def quant_vec(x):
    """dynamic per-vector activation quantization, mirrors q8_vec() in the firmware."""
    x = x.astype(F32)
    amax = np.abs(x).max()
    if amax == 0:
        return np.zeros(x.shape, np.int8), F32(1)
    inv = F32(127.0) / F32(amax)
    s = F32(amax) / F32(127.0)
    q = np.clip(round_away(x * inv), -127, 127).astype(np.int8)
    return q, s

def seqsum(a, axis=-1):
    """sequential float32 sum (like a C for-loop), not numpy's pairwise sum."""
    return np.cumsum(a.astype(F32), axis=axis, dtype=F32).take(-1, axis=axis)

C0, C1, C2, C3, C4 = (F32(1.0000035762786865), F32(0.692969560623169), F32(0.24162131547927856),
                      F32(0.05171773582696915), F32(0.013683983124792576))

def fast_exp(x):
    """mirror of fast_expf() in the firmware: 2^(x*log2e), deg-4 poly + exponent bits."""
    x = np.asarray(x, F32)
    x = np.minimum(np.maximum(x, F32(-87.0)), F32(88.0))
    t = x * F32(1.442695041)
    i = np.trunc(t).astype(np.int32)
    i = i - (t < i.astype(F32)).astype(np.int32)
    f = t - i.astype(F32)
    p = C0 + f * (C1 + f * (C2 + f * (C3 + f * C4)))
    bits = ((i + 127) << 23).astype(np.int32)
    return (p * bits.view(F32)).astype(F32)

class QModel:
    """Everything the firmware stores: int8 weights + float scales + rope table."""
    def __init__(self, m, seq=None):
        self.m = m
        D, H, L, kv = m["dim"], m["hidden"], m["L"], m["kv"]
        self.Hp = Hp = ceil16(H)
        self.seq = seq or m["seq"]
        assert D % 16 == 0 and kv % 16 == 0, "dim and kv_dim must be multiples of 16"
        self.emb_q, self.emb_s, _ = quant_rows(m["emb"])
        self.layers = []
        for l in range(L):
            w1 = np.zeros((Hp, D), F32); w1[:H] = m["w1"][l]
            w3 = np.zeros((Hp, D), F32); w3[:H] = m["w3"][l]
            w2 = np.zeros((D, Hp), F32); w2[:, :H] = m["w2"][l]
            qkv = np.concatenate([m["wq"][l], m["wk"][l], m["wv"][l]])
            lay = dict(rms_att=m["rms_att"][l].astype(F32), rms_ffn=m["rms_ffn"][l].astype(F32))
            for name, w in (("qkv", qkv), ("wo", m["wo"][l]), ("w13", np.concatenate([w1, w3])), ("w2", w2)):
                q, s, _ = quant_rows(w)
                lay[name] = (q, s)
            self.layers.append(lay)
        self.rms_final = m["rms_final"].astype(F32)
        self.rope = rope_table(m, self.seq)

    # ── binary blob for the "model" flash partition ──
    def blob(self):
        m = self.m
        hdr = struct.pack("<4s13I", b"ESPL", 1, m["dim"], m["hidden"], self.Hp, m["L"], m["nh"], m["nkv"],
                          m["vocab"], self.seq, m["kv"], m["hs"], 0, 0)
        out = bytearray(hdr.ljust(64, b"\0"))
        def put(arr):
            out.extend(np.ascontiguousarray(arr).tobytes())
            out.extend(b"\0" * ((-len(out)) % 16))
        put(self.emb_q); put(self.emb_s); put(self.rms_final)
        for lay in self.layers:
            put(lay["rms_att"]); put(lay["rms_ffn"])
            for n in ("qkv", "wo", "w13", "w2"):
                put(lay[n][0]); put(lay[n][1])
        put(self.rope)
        struct.pack_into("<I", out, 56, len(out))
        return bytes(out)

class Q8Sim:
    """
    Bit-level mirror of the firmware's int8 path (modes 1..5). Every float op is
    done in float32 in the same order as the C code, sums are sequential.
    act_quant=False mirrors mode 0 (float32 matmuls on dequantized weights).
    """
    def __init__(self, qm, act_quant=True, int_qk=False):
        self.q, self.aq, self.int_qk = qm, act_quant, int_qk
        m = qm.m
        self.deq = None
        if not act_quant:
            self.deq = {}
        self.reset()

    def reset(self):
        q, m = self.q, self.q.m
        self.k8 = np.zeros((m["L"], q.seq, m["kv"]), np.int8)
        self.v8 = np.zeros_like(self.k8)
        self.ks = np.zeros((m["L"], q.seq), F32)
        self.vs = np.zeros_like(self.ks)

    def matmul(self, W, x):
        wq, ws = W
        if self.aq:
            xq, sx = quant_vec(x)
            acc = wq.astype(np.int32) @ xq.astype(np.int32)
            return (acc.astype(F32) * ws) * sx
        wf = wq.astype(F32) * ws[:, None]
        return seqsum(wf * x[None, :], axis=1)

    def rmsnorm(self, x, w):
        D = x.shape[0]
        ss = seqsum(x * x)
        ss = ss / F32(D) + F32(1e-5)
        ss = F32(1.0) / np.sqrt(F32(ss))
        return (w * (F32(ss) * x)).astype(F32)

    def forward(self, tok, pos):
        q, m = self.q, self.q.m
        D, hs, nh, nkv, kv = m["dim"], m["hs"], m["nh"], m["nkv"], m["kv"]
        Hp, rep = q.Hp, nh // nkv
        x = q.emb_q[tok].astype(F32) * q.emb_s[tok]
        cs = q.rope[pos]
        isq = F32(1.0) / np.sqrt(F32(hs))
        for l, lay in enumerate(q.layers):
            xb = self.rmsnorm(x, lay["rms_att"])
            qkv = self.matmul(lay["qkv"], xb)
            qv = apply_rope(qkv[:D].copy(), nh, hs, cs)
            kvec = apply_rope(qkv[D:D + kv].copy(), nkv, hs, cs)
            vvec = qkv[D + kv:]
            self.k8[l, pos], self.ks[l, pos] = quant_vec(kvec)
            self.v8[l, pos], self.vs[l, pos] = quant_vec(vvec)
            att_out = np.zeros(D, F32)
            for h in range(nh):
                kh = h // rep
                K = self.k8[l, :pos + 1, kh * hs:(kh + 1) * hs].astype(F32)
                V = self.v8[l, :pos + 1, kh * hs:(kh + 1) * hs].astype(F32)
                if self.int_qk:
                    q8, sq = quant_vec(qv[h * hs:(h + 1) * hs])
                    acc = self.k8[l, :pos + 1, kh * hs:(kh + 1) * hs].astype(np.int32) @ q8.astype(np.int32)
                    sc = (acc.astype(F32) * (sq * isq)) * self.ks[l, :pos + 1]
                else:
                    qh = qv[h * hs:(h + 1) * hs] * isq
                    sc = seqsum(K * qh[None, :], axis=1) * self.ks[l, :pos + 1]
                mx = sc.max()
                e = fast_exp(sc - mx)
                inv = F32(1.0) / seqsum(e)
                w = (e * inv) * self.vs[l, :pos + 1]
                att_out[h * hs:(h + 1) * hs] = seqsum(w[:, None] * V, axis=0)
            x = x + self.matmul(lay["wo"], att_out)
            xb = self.rmsnorm(x, lay["rms_ffn"])
            h13 = self.matmul(lay["w13"], xb)
            h1, h3 = h13[:Hp], h13[Hp:]
            hh = (h1 / (F32(1.0) + fast_exp(-h1))) * h3
            x = x + self.matmul(lay["w2"], hh)
        xb = self.rmsnorm(x, q.rms_final)
        return self.matmul((q.emb_q, q.emb_s), xb)

def greedy(model, prompt_ids, steps):
    model.reset()
    out, tok, logits_trace = [], prompt_ids[0], []
    for pos in range(steps):
        lg = model.forward(tok, pos)
        if pos + 1 < len(prompt_ids):
            nxt = prompt_ids[pos + 1]
        else:
            nxt = int(np.argmax(lg))
            srt = np.sort(lg)
            logits_trace.append((float(srt[-1]), float(srt[-1] - srt[-2])))
            out.append(nxt)
            if nxt == 1:
                break
        tok = nxt
    return out, logits_trace

def text_of(tk, prompt_ids, gen):
    ids = prompt_ids + gen
    b = b"".join(tk.decode(ids[i - 1], ids[i]) for i in range(1, len(ids)) if ids[i] != 1)
    return b.decode("utf-8", "replace")

# ─────────────────────────────── analyze ───────────────────────────────

PROMPTS = ["Once upon a time", "Lily and Ben went to the park", "The cat was very hungry", "Tom had a red ball",
           "One day, a little bird", "Sara wanted to bake a cake"]

def cmd_analyze(a):
    m = load_llama2c(a.model)
    tk = Tokenizer(find_tok(a.model, a.tok), m["vocab"])
    print(f"model: dim={m['dim']} hidden={m['hidden']} layers={m['L']} heads={m['nh']}/{m['nkv']} "
          f"vocab={m['vocab']} seq={m['seq']} shared_cls={m['shared']}")
    nparams = sum(m[k].size for k in ("emb", "wq", "wk", "wv", "wo", "w1", "w2", "w3")) + (0 if m["shared"] else m["cls"].size)
    print(f"params: {nparams:,}  → fp32 {nparams*4/1024:.0f} KB, int8 {nparams/1024:.0f} KB, int4 {nparams/2048:.0f} KB")
    fm = FloatModel(m)
    steps = a.steps
    ref = {}
    for p in PROMPTS:
        ids = tk.encode(p)
        ref[p] = (ids, greedy(fm, ids, steps)[0])
    print(f"\nfloat32 sample: {text_of(tk, *ref[PROMPTS[0]])[:160]!r}")

    def score(model):
        """mean steps until first divergence from fp32 greedy text + teacher-forced NLL on that text"""
        div, nll, n = [], 0.0, 0
        for p in PROMPTS:
            ids, g = ref[p]
            gq = greedy(model, ids, steps)[0]
            div.append(next((i for i, (u, v) in enumerate(zip(g, gq)) if u != v), len(g)))
            seq = ids + g
            model.reset()
            for pos in range(len(seq) - 1):
                lg = model.forward(seq[pos], pos).astype(np.float64)
                if pos >= len(ids) - 1:
                    lg -= lg.max(); nll += -(lg[seq[pos + 1]] - np.log(np.exp(lg).sum())); n += 1
        return np.mean(div), nll / n

    def wq_model(**kw):
        mm = dict(m)
        for k in ("emb", "wq", "wk", "wv", "wo", "w1", "w2", "w3"):
            mm[k] = np.stack([quant_rows(w, **kw)[2] for w in m[k]]) if m[k].ndim == 3 else quant_rows(m[k], **kw)[2]
        mm["cls"] = mm["emb"] if m["shared"] else quant_rows(m["cls"], **kw)[2]
        return FloatModel(mm)

    qm = QModel(m)
    rows = [("float32 (reference)", fm),
            ("W8, per-row", wq_model(bits=8)),
            ("W6, per-row", wq_model(bits=6)),
            ("W4, per-row", wq_model(bits=4)),
            ("W4, groups of 16", wq_model(bits=4, group=16)),
            ("W8 + fp32 act (fw mode 0)", Q8Sim(qm, act_quant=False)),
            ("W8A8 + KV8 + fast exp (fw)", Q8Sim(qm)),
            ("  + integer Q·K (fw mode 7)", Q8Sim(qm, int_qk=True))]
    print(f"\n{len(PROMPTS)} prompts x {steps} greedy steps; NLL measured on the fp32 model's own text")
    print(f"  {'variant':30s} {'agree until':>11s} {'NLL':>7s} {'ppl change':>10s}")
    base = None
    for name, mdl in rows:
        d, nll = score(mdl)
        base = base if base is not None else nll
        print(f"  {name:30s} {d:8.1f} st {nll:7.3f} {(math.exp(nll - base) - 1) * 100:+9.1f}%")
    ids, g = ref[PROMPTS[0]]
    print(f"\nfirmware-pipeline sample: {text_of(tk, ids, greedy(Q8Sim(qm), ids, steps)[0])[:160]!r}")
    print(f"blob size: {len(qm.blob())/1024:.1f} KB (seq={qm.seq}); with seq=256: {len(QModel(m, 256).blob())/1024:.1f} KB")

# ─────────────────────────────── export ───────────────────────────────

def cmd_export(a):
    m = load_llama2c(a.model)
    qm = QModel(m, a.seq)
    b = qm.blob()
    open(a.out, "wb").write(b)
    print(f"wrote {a.out}: {len(b)/1024:.1f} KB (seq_len={qm.seq})")

# ─────────────────────────────── device side ───────────────────────────────

MODE_NAMES = {
    0: "fp32 · PSRAM · 1 core · scalar",
    1: "int8 · PSRAM · 1 core · scalar",
    2: "int8 · flash XIP · 1 core · scalar",
    3: "int8 · SRAM · 1 core · scalar",
    4: "int8 · SRAM · 1 core · PIE SIMD",
    5: "int8 · SRAM · 2 cores · PIE SIMD",
    6: "  + attention v2 (GQA pairs)",
    7: "  + integer Q·K in attention",
}

def find_port():
    c = sorted(glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/cu.usbserial*") + glob.glob("/dev/cu.wchusbserial*")
               + glob.glob("/dev/cu.SLAB_USBtoUART*") + glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    if not c:
        sys.exit("no serial port found; pass --port")
    return c[0]

class Device:
    def __init__(self, port, baud=115200):
        import serial
        if "://" in port:                       # e.g. socket://localhost:5555 for QEMU
            self.s = serial.serial_for_url(port, timeout=0.2)
        else:
            self.s = serial.Serial()
            self.s.port, self.s.baudrate, self.s.timeout = port, baud, 0.2
            self.s.dtr = False; self.s.rts = False  # set before open: don't reset the board via DTR/RTS
            self.s.open()
        time.sleep(0.3)
        self.s.reset_input_buffer()

    def lines(self, timeout):
        end, buf = time.time() + timeout, b""
        while time.time() < end:
            buf += self.s.read(4096)
            while b"\n" in buf:
                ln, buf = buf.split(b"\n", 1)
                ln = ln.strip().decode("ascii", "replace")
                if ln.startswith(("@", "!")):
                    yield ln
                    end = time.time() + timeout

    def cmd(self, line, timeout=10):
        self.s.write((line + "\n").encode())
        self.s.flush()
        return self.lines(timeout)

    def info(self):
        for _ in range(3):
            for ln in self.cmd("INFO", 3):
                if ln.startswith("@INFO"):
                    return dict(kv.split("=", 1) for kv in ln.split()[1:])
                if ln.startswith("!"):
                    print("device:", ln)
        sys.exit("device does not answer INFO — is the firmware flashed and the right port selected?")

    def gen(self, mode, ids, steps, verify=False):
        toks, us, extra, end = [], [], [], None
        cmd = f"GEN {mode} {steps} {1 if verify else 0} " + ",".join(map(str, ids))
        for ln in self.cmd(cmd, 30):
            p = ln.split()
            if p[0] == "@T":
                toks.append(int(p[1])); us.append(int(p[2]))
                if verify:
                    extra.append((float(p[3]), float(p[4])))
            elif p[0] == "@END":
                end = dict(kv.split("=", 1) for kv in p[1:]); break
            elif p[0].startswith("!"):
                raise RuntimeError(ln)
        return toks, us, extra, end

def load_for_device(a):
    m = load_llama2c(a.model)
    tk = Tokenizer(find_tok(a.model, a.tok), m["vocab"])
    return m, tk

def cmd_bench(a):
    m, tk = load_for_device(a)
    dev = Device(a.port or find_port(), a.baud)
    info = dev.info()
    print("device:", " ".join(f"{k}={v}" for k, v in info.items()))
    ids = tk.encode(a.prompt)
    modes = [int(x) for x in info.get("modes", "").split(",") if x]
    rows = []
    for mode in modes:
        toks, us, _, end = dev.gen(mode, ids, a.steps)
        gen_us = us[len(ids) - 1:]
        tps = len(gen_us) / (sum(gen_us) / 1e6)
        rows.append((mode, tps, end, toks))
        print(f"  mode {mode}  {MODE_NAMES.get(mode, '?'):36s} {tps:8.1f} tok/s   "
              f"matmul {end['mm']}% attn {end['att']}% other {end['oth']}%")
    base = rows[0][1]
    print("\nspeed-up vs first mode:", ", ".join(f"m{r[0]}×{r[1]/base:.1f}" for r in rows))
    print("\nsample (last mode):", text_of(tk, ids, rows[-1][3][len(ids) - 1:])[:300])
    # tokens/s along the sequence for the fastest mode → shows attention cost growing with context
    toks, us, _, _ = dev.gen(rows[-1][0], ids[:1], a.steps)
    pts = [(i, 1e6 / u) for i, u in enumerate(us)]
    print("\ntok/s vs position (mode %d):" % rows[-1][0], " ".join(f"{i}:{v:.0f}" for i, v in pts[::max(1, len(pts)//10)]))
    if a.json:
        json.dump(dict(info=info, modes=[dict(mode=r[0], tps=r[1], prof=r[2]) for r in rows], pos_curve=pts),
                  open(a.json, "w"), indent=1)
        print("saved", a.json)

def cmd_verify(a):
    m, tk = load_for_device(a)
    dev = Device(a.port or find_port(), a.baud)
    info = dev.info()
    seq = int(info["seq"])
    qm = QModel(m, seq)
    sims = {"f32": Q8Sim(qm, False), "q8": Q8Sim(qm, True), "iqk": Q8Sim(qm, True, int_qk=True)}
    sim_of = lambda mode: "f32" if mode == 0 else "iqk" if mode == 7 else "q8"
    modes = [int(x) for x in info.get("modes", "").split(",") if x]
    ok_all = True
    for p in PROMPTS[:a.prompts]:
        ids = tk.encode(p)
        ref = {}
        for key, sim in sims.items():
            sim.reset(); tok = ids[0]; r = []
            for pos in range(a.steps):
                lg = sim.forward(tok, pos)
                top = int(np.argmax(lg)); srt = np.sort(lg)
                r.append((top, float(srt[-1]), float(srt[-1] - srt[-2])))
                tok = ids[pos + 1] if pos + 1 < len(ids) else top
                if tok == 1: break
            ref[key] = r
        print(f"\nprompt {p!r}")
        for mode in modes:
            toks, us, extra, end = dev.gen(mode, ids, a.steps, verify=True)
            r = ref[sim_of(mode)]
            n = min(len(r), len(toks))
            # compare step by step while the device follows the same path
            first_bad, max_dlogit = None, 0.0
            for i in range(n):
                if i + 1 < len(ids):
                    pass
                dl = abs(extra[i][0] - r[i][1])
                max_dlogit = max(max_dlogit, dl)
                if toks[i] != r[i][0]:
                    first_bad = i; break
            if first_bad is None:
                st = "OK  identical tokens"
            else:
                tie = r[first_bad][2] < 1e-3
                st = f"{'TIE' if tie else 'DIFF'} at step {first_bad} (host margin {r[first_bad][2]:.2e})"
                ok_all &= tie
            print(f"  mode {mode} {MODE_NAMES.get(mode,'?'):36s} {st:42s} max|Δlogit|={max_dlogit:.2e}")
    print("\nRESULT:", "PASS — firmware matches the reference" if ok_all else "FAIL — see DIFF lines above")

def cmd_gen(a):
    m, tk = load_for_device(a)
    dev = Device(a.port or find_port(), a.baud)
    info = dev.info()
    mode = a.mode if a.mode is not None else max(int(x) for x in info["modes"].split(","))
    ids = tk.encode(a.prompt)
    sys.stdout.write(a.prompt); sys.stdout.flush()
    prev, t0 = ids[-1], time.time()
    cmd = f"GEN {mode} {a.steps} 0 " + ",".join(map(str, ids))
    n = 0
    for ln in dev.cmd(cmd, 30):
        p = ln.split()
        if p[0] == "@T":
            pos = n; n += 1
            if pos + 1 < len(ids): continue
            tok = int(p[1])
            if tok == 1: break
            sys.stdout.write(tk.decode(prev, tok).decode("utf-8", "replace")); sys.stdout.flush()
            prev = tok
        elif p[0] == "@END":
            print(f"\n\n[{p[1:]}]"); break

def detect_variant(port):
    """read the chip's eFuse feature list: S3R8 = 8 MB octal PSRAM, S3R2 = 2 MB quad PSRAM."""
    import re
    r = subprocess.run([sys.executable, "-m", "esptool", "--chip", "esp32s3", "-p", port, "chip_id"],
                       capture_output=True, text=True)
    txt = r.stdout + r.stderr
    feat = next((l for l in txt.splitlines() if "Features" in l), "")
    print(feat.strip() or txt[-400:])
    mb = re.search(r"Embedded PSRAM (\d+)MB", feat)
    if mb and int(mb.group(1)) >= 8:
        return "octal"
    return "quad"

def cmd_flash(a):
    port = a.port or find_port()
    blob = os.path.join(HERE, "model.espl")
    m = load_llama2c(a.model or os.path.join(HERE, "stories260K.bin"))
    open(blob, "wb").write(QModel(m, a.seq).blob())
    print(f"exported {blob}")
    if a.build:
        fw = os.path.join(HERE, "firmware")
        v = a.variant or detect_variant(port)
        defaults = "sdkconfig.defaults;" + {"octal": "sdkconfig.octal", "quad": "sdkconfig.quad",
                                             "safe": "sdkconfig.quad;sdkconfig.qemu"}[v]
        subprocess.check_call(["idf.py", "-B", f"build_{v}", "-D", f"SDKCONFIG=sdkconfig_{v}",
                               "-D", f"SDKCONFIG_DEFAULTS={defaults}", "build"], cwd=fw)
        d = os.path.join(fw, f"build_{v}")
        files = [os.path.join(d, "bootloader", "bootloader.bin"), os.path.join(d, "partition_table", "partition-table.bin"),
                 os.path.join(d, "esp_llm.bin")]
    else:
        v = a.variant or detect_variant(port)
        d = os.path.join(HERE, "firmware", "prebuilt", v)
        files = [os.path.join(d, n) for n in ("bootloader.bin", "partition-table.bin", "esp_llm.bin")]
    print(f"firmware variant: {v}")
    args = [sys.executable, "-m", "esptool", "--chip", "esp32s3", "-p", port, "-b", str(a.flash_baud),
            "write_flash", "0x0", files[0], "0x8000", files[1], "0x10000", files[2], "0x110000", blob]
    print(" ".join(args))
    subprocess.check_call(args)
    print("done. next: python esp_llm.py verify")

def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    def common(p, dev=True):
        p.add_argument("--model", default=os.path.join(HERE, "stories260K.bin"))
        p.add_argument("--tok", default=None)
        if dev:
            p.add_argument("--port", default=None)
            p.add_argument("--baud", type=int, default=115200)
    p = sp.add_parser("analyze"); common(p, False); p.add_argument("--steps", type=int, default=64)
    p = sp.add_parser("export"); common(p, False); p.add_argument("--out", default="model.espl"); p.add_argument("--seq", type=int, default=256)
    p = sp.add_parser("flash"); common(p); p.add_argument("--build", action="store_true"); p.add_argument("--seq", type=int, default=256)
    p.add_argument("--variant", choices=["quad", "octal", "safe"], default=None,
                   help="quad/octal = PSRAM type (auto-detected), safe = DIO flash + quad PSRAM if the board won't boot")
    p.add_argument("--flash-baud", type=int, default=921600)
    p.set_defaults(model=None)
    p = sp.add_parser("bench"); common(p); p.add_argument("--steps", type=int, default=128)
    p.add_argument("--prompt", default="Once upon a time"); p.add_argument("--json", default="bench.json")
    p = sp.add_parser("verify"); common(p); p.add_argument("--steps", type=int, default=96); p.add_argument("--prompts", type=int, default=3)
    p = sp.add_parser("gen"); common(p); p.add_argument("--steps", type=int, default=256)
    p.add_argument("--prompt", default="Once upon a time"); p.add_argument("--mode", type=int, default=None)
    a = ap.parse_args()
    {"analyze": cmd_analyze, "export": cmd_export, "flash": cmd_flash, "bench": cmd_bench,
     "verify": cmd_verify, "gen": cmd_gen}[a.cmd](a)

if __name__ == "__main__":
    main()
