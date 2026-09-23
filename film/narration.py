"""Voice the narration for "Through Claude's Eyes" with a local Qwen3-TTS Base model.

Run in the qwen3-tts env:
    conda activate qwen3-tts
    python film/narration.py                                   # sample reference voice
    python film/narration.py --ref-audio me.wav --ref-text "exact words I said in me.wav"
    python film/narration.py --xvec                            # timbre only, ignore ref transcript

Outputs (film/out/voice/):
    line_XX.wav      one clip per narration line (cached; --regen to redo)
    narration.wav    all lines placed on the 62 s film timeline
    narration.srt    matching subtitles
    timing.txt       where each line landed vs. its scene
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out" / "voice"
FPS = 8
SCENES = [("title", 40), ("words", 60), ("ocean", 60), ("picture", 56), ("attention", 56),
          ("next", 60), ("reply", 64), ("window", 56), ("hello", 44)]   # frames, as in the HTML
DEFAULT_REF = HERE / "voice" / "ref_clone.wav"
DEFAULT_REF_TEXT = ("Okay. Yeah. I resent you. I love you. I respect you. "
                    "But you know what? You blew it! And thanks to you.")
LEAD_IN = 0.35      # s after a scene cut before its line starts
MIN_GAP = 0.30      # s of breath between consecutive lines
MAX_TEMPO = 1.12    # never speed a line up more than this to make it fit


def scene_windows():
    out, t = {}, 0
    for name, dur in SCENES:
        out[name] = (t / FPS, (t + dur) / FPS)
        t += dur
    return out, t / FPS


def trim_silence(wav, sr, thresh=0.01, pad=0.05):
    idx = np.where(np.abs(wav) > thresh)[0]
    if len(idx) == 0:
        return wav
    a = max(0, idx[0] - int(pad * sr))
    b = min(len(wav), idx[-1] + int(pad * sr))
    return wav[a:b]


def load_model(path, dtype):
    from qwen_tts import Qwen3TTSModel
    return Qwen3TTSModel.from_pretrained(
        str(path), device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=dtype, attn_implementation="sdpa",   # flash-attn 2 does not support Turing GPUs
    )


def srt_time(s):
    ms = int(round(s * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "Qwen3-TTS-12Hz-0.6B-Base"))
    ap.add_argument("--script", default=str(HERE / "narration.json"))
    ap.add_argument("--ref-audio", default=str(DEFAULT_REF))
    ap.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
    ap.add_argument("--xvec", action="store_true", help="clone timbre only (x-vector); ignores --ref-text")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--regen", action="store_true", help="regenerate cached line clips")
    ap.add_argument("--only", type=int, nargs="*", help="regenerate just these line numbers")
    args = ap.parse_args()

    script = json.loads(Path(args.script).read_text())
    lines = script["lines"]
    OUT.mkdir(parents=True, exist_ok=True)

    todo = [i for i in range(len(lines))
            if args.regen or (args.only and i in args.only) or not (OUT / f"line_{i:02d}.wav").exists()]
    if todo:
        model = load_model(args.model, getattr(torch, args.dtype))
        prompt = model.create_voice_clone_prompt(
            ref_audio=args.ref_audio,
            ref_text=None if args.xvec else args.ref_text,
            x_vector_only_mode=args.xvec,
        )
        for i in todo:
            torch.manual_seed(args.seed + i)
            wavs, sr = model.generate_voice_clone(
                text=lines[i]["text"], language=script.get("language", "English"),
                voice_clone_prompt=prompt,
            )
            wav = trim_silence(np.asarray(wavs[0], dtype=np.float32), sr)
            sf.write(OUT / f"line_{i:02d}.wav", wav, sr)
            print(f"line {i}: {len(wav) / sr:5.2f} s  {lines[i]['text']}")

    # ---- lay the clips on the film timeline
    win, total = scene_windows()
    clips = [sf.read(OUT / f"line_{i:02d}.wav", dtype="float32") for i in range(len(lines))]
    sr = clips[0][1]
    track = np.zeros(int((total + 2) * sr), dtype=np.float32)
    report, srt, cursor = [], [], 0.0
    for i, ((wav, _), line) in enumerate(zip(clips, lines)):
        s0, s1 = win[line["scene"]]
        start = max(s0 + LEAD_IN, cursor + MIN_GAP)
        room = (s1 - 0.15) - start
        tempo = 1.0
        if len(wav) / sr > room > 0:
            tempo = min(MAX_TEMPO, (len(wav) / sr) / room)
            import librosa
            wav = librosa.effects.time_stretch(wav, rate=tempo)
        end = start + len(wav) / sr
        a = int(start * sr)
        track[a:a + len(wav)] += wav[: len(track) - a]
        cursor = end
        spill = max(0.0, end - s1)
        report.append(f"{i}  {line['scene']:<10} scene {s0:5.2f}-{s1:5.2f}s  voice {start:5.2f}-{end:5.2f}s"
                      f"  tempo x{tempo:.2f}" + (f"  SPILLS {spill:.2f}s into next scene" if spill > 0.05 else ""))
        srt.append(f"{i + 1}\n{srt_time(start)} --> {srt_time(end)}\n{line['text']}\n")

    track = track[: int(total * sr)]
    peak = np.max(np.abs(track))
    if peak > 0:
        track *= 0.89 / peak
    sf.write(OUT / "narration.wav", track, sr)
    (OUT / "narration.srt").write_text("\n".join(srt))
    (OUT / "timing.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"wrote {OUT / 'narration.wav'} and narration.srt")


if __name__ == "__main__":
    main()
