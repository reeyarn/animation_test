"""Synthesize the soundtrack for "Through Claude's Eyes" with numpy.

Everything is generated from scratch: a kalimba/music-box score in F major
(96 BPM, so one beat = 5 film frames at 8 fps) plus paper-and-desk foley
cued to the same frame timeline the HTML animation uses.

    python film/soundtrack.py            # writes film/out/soundtrack.wav
"""
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, sosfilt

SR = 44100
FPS = 8
# Scene lengths in frames — must match `scenes` in through-claudes-eyes.html.
SCENE_DURS = [40, 60, 60, 56, 56, 60, 64, 56, 44]
SCENE_NAMES = ["title", "words", "ocean", "picture", "attention",
               "next", "reply", "window", "hello"]

rng = np.random.default_rng(2026)


def starts_from(durs):
    out, t = {}, 0
    for name, d in zip(SCENE_NAMES, durs):
        out[name] = t
        t += d
    return out, t


def ft(frame):
    """Film frame -> seconds."""
    return frame / FPS


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def tvec(dur):
    return np.arange(int(dur * SR)) / SR


def norm(x):
    p = np.max(np.abs(x))
    return x / p if p > 0 else x


def band(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], "bandpass", fs=SR, output="sos"), x)


def lowpass(x, f, order=2):
    return sosfilt(butter(order, f, "lowpass", fs=SR, output="sos"), x)


def noise(dur):
    return rng.standard_normal(int(dur * SR))


class Mix:
    def __init__(self, seconds):
        self.buf = np.zeros((int(seconds * SR) + SR, 2))

    def add(self, sig, t, gain=1.0, pan=0.0):
        """Add a mono signal at time t (s). pan: -1 left .. 1 right (constant power)."""
        i = int(t * SR)
        if i >= len(self.buf) or len(sig) == 0:
            return
        sig = sig[: len(self.buf) - i] * gain
        a = (pan + 1) * np.pi / 4
        self.buf[i:i + len(sig), 0] += sig * np.cos(a)
        self.buf[i:i + len(sig), 1] += sig * np.sin(a)


# ---------------------------------------------------------------- instruments
def kalimba(freq, dur=1.8, bright=1.0):
    t = tvec(dur)
    s = (np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.7)
         + 0.22 * np.sin(2 * np.pi * freq * 2.01 * t) * np.exp(-t / 0.25)
         + 0.30 * bright * np.sin(2 * np.pi * freq * 5.4 * t) * np.exp(-t / 0.04))
    return s * np.minimum(1, t / 0.002)


def music_box(freq, dur=2.6):
    t = tvec(dur)
    s = (np.sin(2 * np.pi * freq * t) * np.exp(-t / 1.1)
         + 0.4 * np.sin(2 * np.pi * freq * 3.0 * t) * np.exp(-t / 0.35)
         + 0.15 * np.sin(2 * np.pi * freq * 6.8 * t) * np.exp(-t / 0.08))
    return s * np.minimum(1, t / 0.001)


def pad(notes, dur, attack=0.9, release=1.2):
    t = tvec(dur + release)
    s = np.zeros_like(t)
    for m in notes:
        f = midi(m)
        for cents in (-4, 4):
            ff = f * 2 ** (cents / 1200)
            s += np.sin(2 * np.pi * ff * t) + 0.18 * np.sin(2 * np.pi * 2 * ff * t)
    env = np.minimum(1, t / attack) * np.clip((dur + release - t) / release, 0, 1)
    trem = 1 + 0.08 * np.sin(2 * np.pi * 0.35 * t)
    return lowpass(s * env * trem, 1800) / (2 * len(notes))


def bass(freq, dur=2.4):
    t = tvec(dur)
    s = np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(2 * np.pi * 2 * freq * t)
    return s * np.minimum(1, t / 0.012) * np.exp(-t / 1.1)


# ---------------------------------------------------------------- foley
def tap(bright=1.0, dur=0.09):
    """A paper piece set down on felt."""
    t = tvec(dur)
    click = band(noise(dur), 1200 * bright, min(9000, 5000 * bright)) * np.exp(-t / 0.007)
    thump = np.sin(2 * np.pi * rng.uniform(110, 160) * t) * np.exp(-t / 0.018)
    return norm(0.7 * norm(click) + 0.5 * thump)


def snip():
    """Scissors: two metallic clicks."""
    def click():
        t = tvec(0.05)
        return (norm(band(noise(0.05), 3000, 12000)) * np.exp(-t / 0.004)
                + 0.35 * np.sin(2 * np.pi * rng.uniform(3800, 4600) * t) * np.exp(-t / 0.012))
    out = np.zeros(int(0.12 * SR))
    c1, c2 = click(), click()
    out[:len(c1)] += c1
    k = int(0.06 * SR)
    out[k:k + len(c2)] += 0.8 * c2
    return norm(out)


def swept(x, f0, f1, chunk=512):
    """Band-pass whose centre glides from f0 to f1 (a whoosh)."""
    out = np.empty_like(x)
    n = len(x)
    zi = None
    for i in range(0, n, chunk):
        fc = f0 * (f1 / f0) ** (i / n)
        sos = butter(2, [fc / 1.6, min(fc * 1.6, SR / 2 - 100)], "bandpass", fs=SR, output="sos")
        if zi is None:
            zi = np.zeros((sos.shape[0], 2))
        out[i:i + chunk], zi = sosfilt(sos, x[i:i + chunk], zi=zi)
    return out


def whoosh(dur=0.45, f0=500, f1=3000):
    t = tvec(dur)
    return norm(swept(noise(dur), f0, f1) * np.sin(np.pi * t / dur) ** 2)


def slide(dur=0.5):
    """Paper dragged across felt: grainy mid noise."""
    t = tvec(dur)
    grain = lowpass(np.abs(noise(dur)), 30)
    env = np.sin(np.pi * t / dur) ** 1.5 * (0.6 + grain / grain.max())
    return norm(band(noise(dur), 400, 3000) * env)


def pluck(freq, dur=1.4, damp=0.994):
    """Karplus–Strong: the red yarn, plucked."""
    n = int(dur * SR)
    N = int(SR / freq)
    y = np.zeros(n)
    y[:N] = rng.uniform(-1, 1, N)
    for i in range(N, n):
        y[i] = damp * 0.5 * (y[i - N] + y[i - N - 1])
    return norm(lowpass(y, 2500))


def tick():
    t = tvec(0.04)
    return norm(np.sin(2 * np.pi * 1900 * t) * np.exp(-t / 0.005)
                + 0.6 * np.sin(2 * np.pi * 820 * t) * np.exp(-t / 0.009))


def flick():
    t = tvec(0.07)
    return norm(band(noise(0.07), 2000, 9000) * np.exp(-t / 0.012)
                + 0.4 * np.sin(2 * np.pi * 180 * t) * np.exp(-t / 0.015))


def thunk():
    t = tvec(0.5)
    return norm(np.sin(2 * np.pi * 68 * t) * np.exp(-t / 0.11)
                + 0.5 * band(noise(0.5), 90, 900) * np.exp(-t / 0.04))


def typekey():
    t = tvec(0.12)
    body = band(noise(0.12), 800, 6000) * np.exp(-t / 0.01)
    bar = 0.5 * np.sin(2 * np.pi * 1350 * t) * np.exp(-t / 0.02)
    return norm(norm(body) + bar + 0.4 * np.sin(2 * np.pi * 90 * t) * np.exp(-t / 0.03))


def sea(dur):
    """Brown-noise surf with slow swells."""
    t = tvec(dur)
    b = np.cumsum(noise(dur))
    b = band(b - np.convolve(b, np.ones(2000) / 2000, mode="same"), 60, 1400)
    swell = 0.55 + 0.45 * np.sin(2 * np.pi * t / 4.2 - 1.2) ** 2
    fade = np.minimum(1, t / 1.5) * np.clip((dur - t) / 1.5, 0, 1)
    return norm(b * swell * fade)


def room(dur):
    """Very low room tone / tape hiss so silence isn't digital zero."""
    return norm(lowpass(noise(dur), 3000)) * 0.5


def reverb_ir(seconds=2.2, tau=0.55):
    t = tvec(seconds)
    ir = np.stack([rng.standard_normal(len(t)), rng.standard_normal(len(t))], 1)
    ir *= np.exp(-t / tau)[:, None]
    ir = np.stack([lowpass(ir[:, 0], 5000), lowpass(ir[:, 1], 5000)], 1)
    return ir / np.sqrt((ir ** 2).sum(0))


def reverb(buf, wet=0.25):
    ir = reverb_ir()
    out = np.stack([fftconvolve(buf[:, c], ir[:, c])[: len(buf)] for c in range(2)], 1)
    return buf + wet * out


# ---------------------------------------------------------------- the score
BPM = 96
BEAT = 60 / BPM          # 0.625 s = 5 frames
BAR = 4 * BEAT
CHORDS = {  # arp tones (ascending), bass, pad
    "F":  ([65, 69, 72, 76, 77], 41, [53, 57, 60, 64]),
    "Am": ([64, 67, 69, 72, 76], 45, [52, 57, 60, 64]),
    "Dm": ([62, 65, 69, 72, 74], 38, [50, 53, 57, 60]),
    "Bb": ([62, 65, 70, 74, 77], 46, [50, 53, 58, 62]),
    "C":  ([64, 67, 72, 74, 79], 48, [52, 55, 60, 64]),
}
PROG = ["F", "Am", "Dm", "Bb", "F", "Am", "Bb", "C"]
ARP = [0, 2, 1, 3, 2, 4, 3, 1]


def score(music, S, total):
    end = ft(total)
    arp_from, arp_to = ft(S["words"]), ft(S["window"] + 40)   # arp stops as the shutters close
    pad_to = ft(S["window"] + 38)
    bar_i = 0
    t = 0.0
    while t < pad_to:
        name = PROG[bar_i % len(PROG)]
        tones, b, pnotes = CHORDS[name]
        music.add(pad(pnotes, min(BAR, pad_to - t) + 0.2), t, 0.55)
        if t >= arp_from - 0.01:
            music.add(bass(midi(b)), t, 0.45)
            music.add(bass(midi(b)), t + 2 * BEAT, 0.28)
            for k, idx in enumerate(ARP):
                nt = t + k * BEAT / 2 + rng.normal(0, 0.006)
                if nt >= arp_to:
                    break
                vel = (0.5 if k % 2 else 0.62) * rng.uniform(0.85, 1.1)
                music.add(kalimba(midi(tones[idx])), nt, vel, pan=(idx - 2) * 0.18)
        else:  # title: sparse music box on beats 1 and 3
            music.add(music_box(midi(tones[0] + 12)), t + 0.05, 0.35, pan=-0.2)
            music.add(music_box(midi(tones[2] + 12)), t + 2 * BEAT, 0.3, pan=0.2)
        t += BAR
        bar_i += 1

    # coda over "hello": a small motif, then a last open chord
    h = ft(S["hello"])
    for k, m in enumerate([77, 81, 84]):
        music.add(music_box(midi(m)), h + 3.0 + k * BEAT, 0.34, pan=-0.25 + 0.25 * k)
    for m in [65, 69, 72, 76]:
        music.add(music_box(midi(m), dur=end - (h + 4.6) + 1), h + 4.6, 0.18)
    music.add(pad([53, 57, 60, 64], end - (h + 4.4), attack=0.4, release=0.8), h + 4.4, 0.35)


def foley(fx, S, qs_len=31):
    QS = "what does the ocean sound like?"
    # 2 · words arrive — each letter lands two frames after it appears
    s = S["words"]
    for i, ch in enumerate(QS):
        if ch != " ":
            fx.add(tap(1.1), ft(s + i + 2), 0.30, pan=-0.7 + 1.4 * i / len(QS))
    for k in range(6):
        fx.add(snip(), ft(s + 32 + k), 0.40, pan=-0.6 + 0.24 * k)
    fx.add(slide(0.9), ft(s + 40), 0.35)
    for k in range(7):
        fx.add(tap(1.6, 0.06), ft(s + 48 + k), 0.18, pan=-0.6 + 0.2 * k)

    # 3 · the ocean, secondhand
    s = S["ocean"]
    fx.add(slide(0.8), ft(s), 0.25)
    fx.add(sea(ft(60) + 1.0), ft(s), 0.32)
    for k in range(12):
        st = s + 10 + 3 * k
        fx.add(whoosh(0.5, 400, 2500), ft(st), 0.20, pan=rng.uniform(-0.6, 0.6))
        fx.add(tap(0.9), ft(st + 5), 0.22, pan=-0.6 + 0.11 * k)

    # 4 · picture drops, is cut, patches fly
    s = S["picture"]
    fx.add(whoosh(1.2, 2500, 300), ft(s), 0.25)
    fx.add(thunk(), ft(s + 11), 0.35)
    fx.add(tap(0.8), ft(s + 11), 0.3)
    for k in range(8):
        fx.add(snip(), ft(s + 13 + k), 0.33, pan=-0.4 + 0.1 * k)
    for k in range(24):
        fx.add(tap(1.3, 0.07), ft(s + 26 + 0.9 * k + 6), 0.14, pan=-0.7 + 1.4 * (k % 12) / 11)

    # 5 · attention — pins pushed in, then the yarn is plucked
    s = S["attention"]
    for k in range(7):
        fx.add(tick(), ft(s) + 0.07 * k, 0.12, pan=-0.7 + 0.23 * k)
    thr = [(1, 0, .3), (2, 1, .3), (3, 2, .4), (4, 3, .8), (3, 0, .25), (5, 4, .6), (4, 0, .3),
           (5, 3, .4), (6, 0, .55), (6, 5, .35), (6, 4, .9), (6, 3, 1)]
    for i, (a, b, w) in enumerate(thr):
        f = midi(52 - round(w * 10))  # stronger thread, lower & louder
        fx.add(pluck(f), ft(s + 4 + 3 * i), 0.12 + 0.28 * w, pan=(a + b - 6) / 7)
    fx.add(tap(), ft(s + 46), 0.3)

    # 6 · next word — deal, spin, pick, drop the rest
    s = S["next"]
    fx.add(tap(), ft(s), 0.2)
    for k in range(7):
        fx.add(flick(), ft(s + 2 + 2 * k), 0.30, pan=-0.75 + 0.25 * k)
    for f in [16, 18, 20, 22, 25, 28, 32]:
        fx.add(tick(), ft(s + f), 0.35)
    fx.add(music_box(midi(84)), ft(s + 32), 0.35)
    fx.add(whoosh(0.9, 1800, 250), ft(s + 36), 0.28)
    fx.add(slide(0.6), ft(s + 36), 0.2)

    # 7 · reply, one word at a time — each word lands on a note of the scale
    s = S["reply"]
    scale = [65, 67, 69, 72, 74, 77, 79, 81, 84, 86, 89]
    for k in range(11):
        if k:
            fx.add(tap(1.4, 0.05), ft(s + 5 * k - 3), 0.07, pan=0.3)   # ghost candidates flutter
            fx.add(tap(), ft(s + 5 * k + 2), 0.28, pan=-0.5 + 0.1 * k)
        fx.add(kalimba(midi(scale[k]), bright=0.6), ft(s + 5 * k + 2), 0.2, pan=-0.5 + 0.1 * k)

    # 8 · the window closes
    s = S["window"]
    fx.add(slide(1.6), ft(s), 0.3, pan=0.4)
    fx.add(slide(1.6), ft(s + 4), 0.25, pan=0.5)
    fx.add(slide(1.4), ft(s + 28), 0.3)
    fx.add(thunk(), ft(s + 40), 0.6)
    fx.add(tap(), ft(s + 44), 0.25)

    # 9 · hello — typed
    s = S["hello"]
    for i in range(6):
        fx.add(typekey(), ft(s + 6 + 3 * i), 0.35, pan=-0.2 + 0.08 * i)


def build(out_path, durs=SCENE_DURS):
    S, total = starts_from(durs)
    seconds = ft(total)
    music, fx, amb = Mix(seconds), Mix(seconds), Mix(seconds)
    score(music, S, total)
    foley(fx, S)
    amb.add(room(seconds), 0, 0.006)
    # a faint shutter click on every film frame — the camera taking each shot
    clk = tick()
    for n in range(total):
        amb.add(clk, ft(n), 0.012)

    mix = 0.55 * reverb(music.buf, 0.30) + reverb(fx.buf, 0.12) + amb.buf
    mix = mix[: int(seconds * SR)]
    fade = int(1.2 * SR)
    mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
    mix *= 0.89 / np.max(np.abs(mix))           # peak at -1 dBFS
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(out_path, SR, (mix * 32767).astype(np.int16))
    return out_path, seconds


if __name__ == "__main__":
    p, sec = build(Path(__file__).parent / "out" / "soundtrack.wav")
    print(f"wrote {p} ({sec:.2f} s)")
