"""Background score for "Two Channels", synthesized with numpy.

Light and joyful: D major, 104 BPM. Bouncy marimba off-beat chords, a plucked
ukulele-like strum, glockenspiel tunes, shaker and soft claps. No minor chords;
the jam is played as comic tip-toe pizzicato, the question as a playful tick-tock,
the MD&A and mosaic as the happiest, fullest sections. Soft cues sit on the
visible actions (stamp, bonks, tiles landing, pen). Timing comes from timing.js.

    python music.py          # writes out/music.wav
"""
import json
from functools import lru_cache
import re
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, sosfilt

HERE = Path(__file__).resolve().parent
SR = 48000
BPM = 104
BEAT = 60 / BPM
rng = np.random.default_rng(11)

TIMING = json.loads(re.search(r"const TIMING = (\[.*\]);", (HERE / "timing.js").read_text(), re.S).group(1))
START, _t = {}, 0.0
for _s in TIMING:
    START[_s["id"]] = _t
    _t += _s["dur"]
TOTAL = _t
SC = {s["id"]: s for s in TIMING}


def line_at(scene, k, f):
    ln = SC[scene]["lines"][k]
    return START[scene] + ln["start"] + (ln["end"] - ln["start"]) * f


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def tv(d):
    return np.arange(int(d * SR)) / SR


def lowpass(x, f):
    return sosfilt(butter(2, f, "lowpass", fs=SR, output="sos"), x)


def highpass(x, f):
    return sosfilt(butter(2, f, "highpass", fs=SR, output="sos"), x)


def bandpass(x, lo, hi):
    return sosfilt(butter(2, [lo, hi], "bandpass", fs=SR, output="sos"), x)


# ---------------------------------------------------------------- instruments
def marimba(m, dur=0.7, vel=1.0):
    f, t = hz(m), tv(dur)
    x = np.sin(2 * np.pi * f * t) * np.exp(-t * 7) + 0.25 * np.sin(2 * np.pi * f * 4 * t) * np.exp(-t * 22)
    return x * np.minimum(1, t / 0.002) * 0.22 * vel


def glock(m, dur=1.4, vel=1.0):
    f, t = hz(m), tv(dur)
    x = (np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t * 8)
         + 0.2 * np.sin(2 * np.pi * f * 5.4 * t) * np.exp(-t * 14))
    return x * np.exp(-t * 3.0) * np.minimum(1, t / 0.001) * 0.16 * vel


def pluck(m, dur=0.9, vel=1.0, damp=0.996):
    """Karplus-Strong string: a bright ukulele-ish pluck (cached per pitch/length)."""
    return _pluck(m, dur, damp) * vel


@lru_cache(maxsize=None)
def _pluck(m, dur, damp):
    f = hz(m)
    vel = 1.0
    p = max(2, int(SR / f))
    buf = rng.uniform(-1, 1, p)
    out = np.zeros(int(dur * SR))
    for i in range(len(out)):
        out[i] = buf[i % p]
        buf[i % p] = damp * 0.5 * (buf[i % p] + buf[(i + 1) % p])
    return lowpass(out, 5000) * 0.18 * vel


def pizz(m, vel=1.0):
    f, t = hz(m), tv(0.35)
    x = sum(a * np.sin(2 * np.pi * f * k * t) for k, a in enumerate([1, .5, .25], 1))
    return x * np.exp(-t * 16) * np.minimum(1, t / 0.003) * 0.2 * vel


def bass(m, dur=0.45, vel=1.0):
    f, t = hz(m), tv(dur)
    x = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t) * np.exp(-t * 8)
    return x * np.exp(-t * 5) * np.minimum(1, t / 0.005) * 0.3 * vel


def pad(ms, dur):
    t = tv(dur + 0.8)
    x = sum(np.sin(2 * np.pi * hz(m) * t) for m in ms)
    env = np.minimum(1, t / 0.6) * np.where(t > dur, np.exp(-(t - dur) / 0.25), 1)
    return lowpass(x * env, 1500) * 0.018


def shaker(vel=1.0):
    t = tv(0.07)
    return highpass(rng.standard_normal(len(t)), 5000) * np.exp(-t * 60) * 0.045 * vel


def clap(vel=1.0):
    t = tv(0.18)
    n = bandpass(rng.standard_normal(len(t)), 900, 3500)
    env = np.exp(-t * 28) * (1 + 0.6 * (np.abs(t - 0.012) < 0.004) + 0.4 * (np.abs(t - 0.024) < 0.004))
    return n * env * 0.12 * vel


def kick(vel=1.0):
    t = tv(0.3)
    f = 110 * np.exp(-t * 25) + 50
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 14) * 0.35 * vel


def boing(vel=1.0):
    """A cartoon bonk for the jam: a pitch-bent spring over a soft thump."""
    t = tv(0.55)
    f = 180 * (1 + 0.35 * np.sin(2 * np.pi * 9 * t) * np.exp(-t * 5)) * np.exp(-t * 1.2)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 5) * 0.3
    k = kick(.8)
    return (x + np.pad(k, (0, len(t) - len(k)))) * vel


def stamp():
    t = tv(0.3)
    return lowpass(rng.standard_normal(len(t)), 1500) * np.exp(-t * 30) * 0.35 + np.sin(2 * np.pi * 180 * t) * np.exp(-t * 20) * 0.25


def scratch(dur=0.3):
    t = tv(dur)
    n = bandpass(rng.standard_normal(len(t)), 2500, 7000)
    return n * (0.5 + 0.5 * np.sin(2 * np.pi * 11 * t) ** 2) * np.sin(np.pi * t / dur) * 0.015


def reverb(x, secs=1.4, wet=0.18):
    t = tv(secs)
    ir = lowpass(rng.standard_normal(len(t)) * np.exp(-t / 0.28), 6000)
    ir /= np.sqrt(np.sum(ir ** 2))
    return x * (1 - wet) + fftconvolve(x, ir)[: len(x)] * wet


# ---------------------------------------------------------------- harmony: major only (I IV V vi), one chord per bar
D, G, A, Bm, Em, Dsus = [50, 57, 62, 66], [43, 55, 59, 62], [45, 57, 61, 64], [47, 54, 59, 62], [40, 55, 59, 64], [50, 57, 62, 67]
PLAN = {   # scene: (chord cycle, texture)
    "title":    ([D, G, A, D], "intro"),
    "channels": ([D, G, D, A], "bounce"),
    "fits":     ([D, G, A, D], "happy"),
    "stuck":    ([G, D, G, A], "tiptoe"),
    "question": ([Em, A, Em, A], "tiptoe"),
    "mdna":     ([G, A, D, Bm], "happy"),
    "numbers":  ([D, G, D, A], "bounce"),
    "methods":  ([G, D, A, D], "bounce"),
    "mosaic":   ([G, A, D, D], "happy"),
    "end":      ([G, A, D, D], "outro"),
}
# glockenspiel tunes in scale steps of D major, 8 eighth-notes per bar (None = rest)
SCALE = [62, 64, 66, 67, 69, 71, 73, 74, 76, 78, 79, 81]
TUNES = [
    [2, None, 4, None, 7, None, 4, 5],
    [4, None, 2, None, 0, None, None, None],
    [2, 4, 5, 7, None, 7, 5, 4],
    [5, None, 4, None, 2, None, None, None],
]


def build():
    n = int((TOTAL + 3) * SR)
    mus, fx = np.zeros(n), np.zeros(n)

    def put(buf, t0, x, gain=1.0):
        a = int(t0 * SR)
        if 0 <= a < n:
            x = x[: n - a]
            buf[a:a + len(x)] += x * gain

    bar, e = 4 * BEAT, BEAT / 2
    tune_i = 0
    for s in TIMING:
        sid, s0, dur = s["id"], START[s["id"]], s["dur"]
        chords, tex = PLAN[sid]
        end = s0 + dur - 0.15
        b = 0
        while s0 + b * bar < end - 0.3:
            tb = s0 + b * bar
            ch = chords[b % len(chords)]
            put(mus, tb, pad(ch[1:], min(bar, end - tb)))
            for k in range(8):                        # eighth-note grid
                tn = tb + k * e
                if tn > end:
                    break
                if tex in ("bounce", "happy"):
                    if k in (0, 4):
                        put(mus, tn, bass(ch[0] + (7 if k == 4 and tex == "happy" else 0)))
                    if k in (2, 6):                   # off-beat marimba chord
                        for m in ch[1:]:
                            put(mus, tn, marimba(m + 12, vel=.55))
                    put(mus, tn, shaker(vel=1.0 if k % 2 else .55))
                    if tex == "happy" and k in (2, 6):
                        put(mus, tn, clap(vel=.8))
                    if tex == "happy" and k in (0, 3, 5):        # ukulele strum
                        for j, m in enumerate(ch[1:]):
                            put(mus, tn + j * 0.012, pluck(m + 12, vel=.45))
                elif tex == "tiptoe":                 # comic pizzicato on the beats, glock sparkles
                    if k % 2 == 0:
                        m = ch[1:][(k // 2) % 3] + (0 if k < 4 else 12)
                        put(mus, tn, pizz(m, vel=.9))
                    if k == 0:
                        put(mus, tn, bass(ch[0], vel=.7))
                elif tex in ("intro", "outro"):       # plucked arpeggio, no drums
                    arp = ch[1:] + [ch[2] + 12]
                    put(mus, tn, pluck(arp[k % 4] + 12, vel=.5))
                    if k == 0:
                        put(mus, tn, bass(ch[0], .9, .8))
            if tex in ("bounce", "happy", "intro", "outro"):     # the tune
                tune = TUNES[tune_i % len(TUNES)]
                tune_i += 1
                if tex != "bounce" or b % 2 == 0:
                    for k, st in enumerate(tune):
                        tn = tb + k * e
                        if st is not None and tn < end - .2:
                            put(mus, tn, glock(SCALE[st] + (12 if tex == "happy" else 0), vel=.8 if tex == "happy" else .65))
            b += 1

    # the final chord rings out bright
    tf = START["end"] + SC["end"]["dur"] - 3.2
    for j, m in enumerate([62, 66, 69, 74, 78]):
        put(mus, tf + j * .05, glock(m + 12, 3.0, .7))
        put(mus, tf + j * .02, pluck(m, 2.5, .5))

    # ---- cues on the visible actions (times mirror two-channels.html)
    put(fx, line_at("channels", 0, .5), stamp())
    for k in range(12):
        put(fx, START["fits"] + 3.0 + k * .12, glock(SCALE[min(11, k)] + 12, vel=.55))
    for tb in (2.6, 3.7, 4.8):
        put(fx, START["stuck"] + tb, boing(), .9)
    for k, tb in enumerate((3.6, 4.3, 5.0)):
        put(fx, START["stuck"] + tb, glock(SCALE[4 + 2 * k] + 12, vel=.5))
    put(fx, START["stuck"] + 6.1, kick(.7))                   # the blob lands on the floor
    l0, l1 = SC["mdna"]["lines"]
    card_t = [l0["start"] + x for x in (1.0, 2.2, 3.4)] + [l1["start"] + x for x in (.2, 1.2, 2.2, 3.1, 3.9, 4.6)]
    for tb in np.arange(0.0, l1["end"] + .2, .5):
        put(fx, START["mdna"] + tb, scratch())
    for k, tc in enumerate(card_t):
        put(fx, START["mdna"] + tc + 2.6 + .55, glock(SCALE[(k * 2) % 12] + 12, vel=.55))

    out = reverb(mus) + reverb(fx, wet=.12)
    end = int(TOTAL * SR)
    out = out[:end]
    fi, fo = int(0.6 * SR), int(2.5 * SR)
    out[:fi] *= np.linspace(0, 1, fi)
    out[-fo:] *= np.linspace(1, 0, fo)
    out = out / np.max(np.abs(out)) * 0.8
    stereo = np.stack([out, np.roll(out, int(.009 * SR)) * .9 + out * .1], axis=1)
    (HERE / "out").mkdir(exist_ok=True)
    wavfile.write(HERE / "out" / "music.wav", SR, (stereo * 32767).astype(np.int16))
    print(f"out/music.wav  {TOTAL:.2f} s")


if __name__ == "__main__":
    build()
