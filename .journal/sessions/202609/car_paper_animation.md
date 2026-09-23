--- [7] PROMPT ---
/model claude-opus-5-5

--- [11] PROMPT ---
/hand-drawn-canvas-animation create a video animation for this paper's idea of two disclosure channels

Financial statement adequacy and firms’ MD&A disclosures
Stephen V. Brown1 | Lisa A. Hinson2 | Jennifer Wu Tucker2


Abstract
Firms are required to provide financial information via the
financial statements and the management discussion and
analysis (MD&A), a narrative explanation of the financial
statements. Our study examines how firms use the MD&A
channel when their financial statement channel is inade-
quate. We focus on two textual attributes of the MD&A:
non-GAAP disclosure and forward-looking statements.
We find that firms with less adequate financial statements
discuss non-GAAP measures more and provide a larger
number of forward-looking statements. We then identify
the topics, and therefore the context, in which non-GAAP
and forward-looking disclosures are provided. Our study
provides evidence on how managers use the MD&A, a rel-
atively more flexible channel, to provide information when
their financial statement channel is less adequate.
KEYWORDS
deep learning, disclosure, machine learning, MD&A, text segmentation,
topic analysis

1 | INTRODUCTION
Firms are required to provide financial information to the public via two channels in their
reports filed with the SEC. The financial statement channel includes the balance sheet, income
statement, cash flow statement, statement of shareholders’ equity, and the accompanying notes
(hereinafter,“financial statements”), all presented in Item 8 of an annual report.1 This channel
is regulated by GAAP, which has specific rules for almost every financial statement item, and is
audited for compliance with GAAP. The management discussion and analysis (MD&A) chan-
nel includes disclosure about liquidity and capital resources, results of operations, and other
issues required by the SEC, as well as voluntary disclosure, all presented in Item 7 of the annual
report. The MD&A is intended to“provide a narrative explanation of a company’s financial
statements and other statistical data” and“provide management with flexibility to describe the
financial matters impacting the registrant” (SEC, 2008, emphasis added). Our study examines
how managers use the MD&A channel when their financial statement channel is less adequate.2
This research question is important for two reasons. First, concerns about the adequacy of
GAAP are widespread. Critics argue that accounting rules are more appropriate for a tradi-
tional fixed-assets-based economy and less adequate for a knowledge-based economy (Amir &
Lev, 1996; S. Brown et al., 1999; Lev, 1989; Lev & Zarowin, 1999; Srivastava, 2014). As a
result, the financial statements prepared under GAAP by some firms may not adequately com-
municate information about these firms’ financial positions, operating results, and future pros-
pects to investors. It is thus critical to investigate what managers do when their financial
statement channel is constrained by GAAP.
Second, the MD&A is arguably the most widely read section of a financial report beyond
the financial statement section (Li, 2010; Tavcar, 1998). Financial analysts acknowledge that
they most frequently rely upon the MD&A over other disclosure sections of an annual report
when performing their tasks (Knutson, 1993; Rogers & Grant, 1997). The MD&A was man-
dated by the SEC via an amendment to Regulation S-K in 1980 with three objectives: (1) provid-
ing a narrative explanation of a firm’s financial statements that enables investors to see the firm
through the eyes of management, (2) providing the context within which financial information
should be analyzed, and (3) providing information to assist investors in predicting future perfor-
mance (SEC, 2008, 2013). The SEC has conducted several reviews of MD&A practices and
1We use“financial statements” to refer to the four financial statements and their accompanying notes because the notes are (1) an
integral part of the financial statements and (2) prepared under GAAP.
2We focus on the financial statements and the MD&A channels because both are mandatory and presented back-to-back in an annual
report. Managers have other channels to communicate financial information with the public, for example, press releases and conference
calls. Examining other channels is outside of the scope of this study.

issued multiple interpretive releases, suggesting a significant emphasis on the MD&A. Our
evidence sheds light on how firms use this important communication channel.
If managers follow the spirit of MD&A regulation, they are expected to provide more infor-
mation in the MD&A when the financial statement channel cannot adequately communicate
financial information to investors. When the financial statement channel is in fact adequate,
managers can save the effort of using the MD&A to communicate with investors. These expec-
tations are consistent with the theoretical prediction of Einhorn (2005), who demonstrates that
when one communication channel is constrained, managers make more use of another, less-
constrained communication channel.
The above expectations may not materialize for two reasons. First, managers might view
financial reporting as a compliance task and choose straightforward compliance over providing
information in the best possible way (Dichev et al., 2013). For example, one CFO said,
“Because at the end of the day, how should I spend my time? Do I want to spend my time work-
ing on this? Or do I want to spend my time working on strategy and driving the business? We’re
not going to let the accounting wag the business here, so we’re just going to comply” (Dichev
et al., 2013, pp. 23–24). In a compliance mindset, managers will meet the minimum disclosure
requirements irrespective of the adequacy of the financial statement channel and save their
firms’ resources for other activities. Second, managers might exercise their discretion opportu-
nistically. Prior research finds that when firms report lower earnings or manipulate earnings,
the MD&A is less readable (Li, 2008; Lo et al., 2017). Opportunistic managers are unlikely to
prioritize providing information to investors in the best possible way.
Our primary proxy for the adequacy of the financial statement channel is the value rele-
vance of earnings and book value of equity.“Value relevance” is defined as the ability of a
firm’s financial statements to capture or summarize information that affects the firm’s stock
value (Francis & Schipper, 1999). A large body of research uses the explanatory power of earn-
ings and book value of equity for stock price or the association of earnings with stock price
(or their respective changes forms) to measure value relevance (Barth et al., 2001). Following
Banker et al. (2009), we use the R2 from regressing a firm’s stock price on its EPS and book
value of equity per share in a time series as the measure of value relevance for the sample year.
We refer to one minus this measure as“financial statement inadequacy.” In supplementary ana-
lyses, we use alternative proxies constructed from a firm’s information in the sample year alone.
We focus on two textual attributes of the MD&A channel. The first is the intensity of non-
GAAP disclosure. If managers intend to provide context within which their financial statements
should be analyzed, as stated in the second regulatory objective of the MD&A, managers could
discuss non-GAAP measures, which are customized to their firm’s situation. In other words,
non-GAAP disclosure encapsulates the firm’s performance that managers tailor to reflect the
firm’s particular situation. The second attribute is the number of forward-looking statements
(FLS). If the financial statements, which summarize historical transactions, are inadequate for
helping investors project future performance, managers may provide FLS to bridge the infor-
mation gap and achieve the third regulatory objective of the MD&A.
We use two textual analysis approaches to measure MD&A textual attributes. To identify
non-GAAP disclosure in the MD&A, we use the traditional keyword approach, which is simple
and relatively accurate for this task (Jo & Yang, 2020). To classify FLS, we use an advanced
approach because it performs substantially better than the keyword approach. Specifically, we
use a deep learning model, implemented with a convolutional neural network (CNN), to identify
FLS as well as the FLS that contain quantitative information (see Bochkay et al., 2023 for a review
of deep learning models). We train the machine to recognize patterns of consecutive words,
numbers, and symbols for predefined categories in a small hand-coded sample and then let the
machine identify and classify unseen phrases and sentences in our full sample. Compared with the
keyword approach, the deep learning approach can learn subtle relationships that the keyword
approach cannot and is robust to coding errors as long as the training sample is large enough.
19113846, 2024, 1, Downloaded from https://onlinelibrary.wiley.com/doi/10.1111/1911-3846.12919 by University Of North Carolina Chapel Hill, Wiley Online Library on [02/02/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License
FINANCIAL STATEMENTS AND MD&A DISCLOSURES 129
Using firm-years ended in 2004–2016, we document several findings. First, firms with less
adequate financial statements discuss non-GAAP measures more in the MD&A. The propor-
tional increase in the intensity of non-GAAP disclosure from firms in the most adequate group
to firms in the least adequate group is 19.0%. This result suggests that firms with a less adequate
financial statement channel turn to the MD&A channel to customize financial information for
investors.3
Our second finding is that firms with less adequate financial statements provide more FLS.
On average, our sample firms provide 30 forward-looking sentences in the MD&A, including
12 forward-looking sentences that contain quantitative and therefore hard information. From
the most adequate group to the least adequate group, the number of FLS increases by 1.508
and the number of quantitative FLS increases by 0.613, equivalent to 5.1% and 4.9% of the
unconditional intensity, respectively. These results suggest that firms with less adequate finan-
cial statements provide more forward-looking information for investors to project the future.
We then identify the topics, and therefore the context, in which non-GAAP and forward-
looking disclosures appear. Using TopicTiling, a combination of topic modeling and text
segmentation, we split an MD&A document into text segments and assign each text segment
(i.e., consecutive sentences that discuss the same topic) a latent topic. Our topic model discovers
133 topics, which are sorted into 21 categories and then aggregated into four groups: required,
traditional, intangibles, and other. We find that firms with less adequate financial statements
discuss non-GAAP measures more in required topics, traditional topics, and intangibles topics
and provide more FLS in required topics and intangibles topics. These results suggest that firms
provide additional disclosures in areas where traditional financial reporting is likely the
weakest.4
Our study contributes to the accounting literature by emphasizing what managers do when
the boundaries of GAAP constrain the provision of adequate financial statement information
to investors. Amir and Lev (1996) examine the wireless communications industry and find that
earnings and book value of equity do not explain a firm’s market value; however, the authors
find that firms provide nonfinancial information to help investors. Our study extends this
research by reporting that managers also use the MD&A to address financial statement defi-
ciencies. Lang and Lundholm (1993) examine analysts’ ratings of a firm’s overall disclosure,
which includes annual reports, quarterly reports, press releases, and investor relations. They
find higher ratings for firms with weaker earnings-return correlations, suggesting that firms pro-
vide additional disclosure when their financial statements are inadequate. We provide direct evi-
dence on the specific types of corporate disclosures supplied when financial statements are
inadequate.
From a broader perspective, managers have a breadth of channels to provide information
to investors. It is unclear whether managers adopt a mosaic approach, characterized by using
multiple channels in combination, or a fragmented approach, characterized by isolated compli-
ance with individual disclosure requirements. Few studies examine more than one managerial
decision at a time and even fewer studies examine a less flexible channel in combination with a
more flexible channel. Our findings are consistent with the mosaic approach.
3Given the paucity of evidence in the literature about non-GAAP disclosure in the MD&A, we hand-coded 200 firms randomly selected
from those that provide non-GAAP disclosure in both the earnings announcement press release and the subsequent MD&A. We observe
that firms provide a variety of non-GAAP measures beyond the traditionally examined non-GAAP EPS and that non-GAAP disclosure
is sometimes more extensive in the earnings announcement than in the MD&A. Even if a non-GAAP measure has appeared in the
earnings announcement, its appearance in the subsequent MD&A is meaningful because the latter is final and placed in close proximity
to the audited financial statements.
4We also find significant differences in another MD&A textual attribute—the total amount of new information in a firm’s MD&A. We
measure the amount of new information using the degree of year-over-year MD&A modifications (S. V. Brown & Tucker, 2011). Firms
with less adequate financial statements modify their MD&As to a greater extent and therefore provide more new information in
the MD&A.
19113846, 2024, 1, Downloaded from https://onlinelibrary.wiley.com/doi/10.1111/1911-3846.12919 by University Of North Carolina Chapel Hill, Wiley Online Library on [02/02/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License
130 CONTEMPORARY ACCOUNTING RESEARCH
Our study is related to Hribar et al. (2022). Hribar et al. proxy for managers’ constraints in
GAAP reporting by the frequency of“shall,”“should,” and“must” in the applicable accounting
standards. The authors examine the likelihood of non-GAAP EPS in earnings announcements
and the frequency of management earnings forecasts—voluntary disclosure channels that we do
not examine. The authors examine the MD&A but limit their examination to readability. Their
study and our study are complementary in that the two studies examine different channels yet
both conclude that managers use more flexible communication channels when their GAAP
channel is constrained.
Our study also contributes to the textual analysis literature. So far, the literature has evolved
in three phases. In the first phase, researchers analyzed simple textual attributes, such as length,
tone, and readability, of narrative information available to market participants (Henry &
Leone, 2016; Li, 2008). In the second phase, techniques were introduced to compare document
similarity either from year to year within a firm or between a firm and its benchmark firms
(S. V. Brown & Knechel, 2016; S. V. Brown & Tucker, 2011; Hoberg & Phillips, 2010; Peterson
et al., 2015). In the third phase, researchers use machine learning approaches to capture more
nuanced textual attributes (Campbell et al., 2014; N. C. Brown et al., 2020; Dyer et al., 2017;
Frankel et al., 2016; Frankel et al., 2022; Huang et al., 2018; Li, 2010). We are one of the first
in accounting and finance to use deep learning—an advanced form of machine learning—in tex-
tual analysis. Deep learning models are built on neural networks and can discover intricate
structures in large data (Bochkay et al., 2023; LeCun et al., 2015). In addition, we introduce
TopicTiling as an advanced technique for topic analysis. These tools can help researchers
address a wide range of questions in the future.

--- [68] ANSWER ---
I'm working through the animation skill's references and examples first. Next I'll write the brief and the narration, then build the film.

--- [106] ANSWER ---
Still in setup. No handwriting font is installed, so I'm fetching Caveat (OFL licence) and embedding it so the lettering renders the same every time.

--- [125] ANSWER ---
Engine files are copied and the renderer is installed. Now I'm writing the narration script and a voice script. I'll generate the voice first, then time each scene to the spoken lines.

--- [126] EDIT:Write ---
[Write /home/hofmannp/animation_test/disclosure-channels/narration.json]
{
  "language": "English",
  "note": "text = subtitle; say = what the TTS model reads (phonetic where needed). tail = seconds held after the last line.",
  "scenes": [
    {"id": "title", "lead": 1.4, "tail": 1.0, "lines": [
      {"text": "Every year, a public firm tells investors how it is doing, through two channels in one annual report.",
       "say":  "Every year, a public firm tells investors how it is doing, through two channels, in one annual report."}
    ]},
    {"id": "channels", "lead": 0.6, "tail": 1.0, "lines": [
      {"text": "Item 8, the financial statements, is strict. GAAP has a rule for almost every line, and auditors check it.",
       "say":  "Item eight, the financial statements, is strict. Gap has a rule for almost every line, and auditors check it."},
      {"text": "Item 7, the MD&A, is management's own narrative. It is far more flexible.",
       "say":  "Item seven, the M D and A, is management's own story. It is far more flexible."}
    ]},
    {"id": "fits", "lead": 0.5, "tail": 1.2, "lines": [
      {"text": "For a factory-style firm, the statements work well: earnings and book value explain the stock price.",
       "say":  "For a factory-style firm, the statements work well. Earnings and book value explain the stock price."}
    ]},
    {"id": "stuck", "lead": 0.6, "tail": 0.8, "lines": [
      {"text": "But value built on ideas, software and brands doesn't fit GAAP's boxes. Price drifts away from the numbers.",
       "say":  "But value built on ideas, software and brands doesn't fit gap's boxes. The price drifts away from the numbers."},
      {"text": "The authors call this financial statement inadequacy: one minus that R-squared.",
       "say":  "The authors call this, financial statement inadequacy. One, minus that R squared."}
    ]},
    {"id": "question", "lead": 0.4, "tail": 0.9, "lines": [
      {"text": "So what does the manager do? Just comply and move on, or turn to the flexible channel?",
       "say":  "So what does the manager do? Just comply, and move on? Or turn to the flexible channel?"}
    ]},
    {"id": "mdna", "lead": 0.5, "tail": 1.2, "lines": [
      {"text": "They turn. Firms with weaker statements discuss non-GAAP measures more, tailored to their own situation.",
       "say":  "They turn. Firms with weaker statements discuss non-gap measures more, tailored to their own situation."},
      {"text": "And they write more forward-looking statements, so investors can see where the firm is heading.",
       "say":  "And they write more forward-looking statements, so investors can see where the firm is heading."}
    ]},
    {"id": "numbers", "lead": 0.5, "tail": 1.6, "lines": [
      {"text": "From the most to the least adequate firms, non-GAAP discussion rises by 19 percent,",
       "say":  "From the most, to the least adequate firms, non-gap discussion rises by nineteen percent,"},
      {"text": "and forward-looking sentences rise by about 5 percent.",
       "say":  "and forward-looking sentences rise by about five percent."}
    ]},
    {"id": "methods", "lead": 0.5, "tail": 1.0, "lines": [
      {"text": "The MD&As are read by machine. Keywords catch non-GAAP talk; a deep-learning network spots forward-looking sentences.",
       "say":  "The M D and A's are read by machine. Keywords catch non-gap talk. A deep-learning network spots forward-looking sentences."},
      {"text": "TopicTiling cuts each document into topics. The extra disclosure appears where traditional reporting is weakest, such as intangibles.",
       "say":  "Topic tiling cuts each document into topics. The extra disclosure appears where traditional reporting is weakest, such as intangibles."}
    ]},
    {"id": "mosaic", "lead": 0.5, "tail": 1.4, "lines": [
      {"text": "So managers use their channels together, like a mosaic. When one is constrained, the other carries more.",
       "say":  "So managers use their channels together, like a mosaic. When one is constrained, the other carries more."}
    ]},
    {"id": "end", "lead": 0.6, "tail": 3.2, "lines": [
      {"text": "Brown, Hinson and Tucker, Contemporary Accounting Research, 2024.",
       "say":  "Brown, Hinson, and Tucker. Contemporary Accounting Research. Twenty twenty-four."}
    ]}
  ]
}


--- [135] EDIT:Write ---
[Write /home/hofmannp/animation_test/disclosure-channels/voice.py]
"""Voice the narration with the local Qwen3-TTS Base model (x-vector clone) and
derive the film's scene timing from the spoken lines.

    conda activate qwen3-tts
    python voice.py              # generate missing clips, then lay out the timeline
    python voice.py --regen      # regenerate every clip
    python voice.py --only 3 7   # regenerate clips 3 and 7 (global line numbers)

Writes out/voice/line_XX.wav, out/voice/narration.wav, out/voice/narration.srt
and timing.js, which the film loads: every scene lasts lead + lines + gaps + tail.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out" / "voice"
FPS = 24
GAP = 0.45          # s between lines inside a scene
REF = ROOT / "film" / "voice" / "ref_clone.wav"


def trim(wav, sr, thresh=0.01, pad=0.05):
    idx = np.where(np.abs(wav) > thresh)[0]
    return wav if not len(idx) else wav[max(0, idx[0] - int(pad * sr)): idx[-1] + int(pad * sr)]


def srt_time(s):
    ms = int(round(s * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "Qwen3-TTS-12Hz-0.6B-Base"))
    ap.add_argument("--regen", action="store_true")
    ap.add_argument("--only", type=int, nargs="*")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    script = json.loads((HERE / "narration.json").read_text())
    lines = [ln for sc in script["scenes"] for ln in sc["lines"]]
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [k for k in range(len(lines)) if args.regen or (args.only and k in args.only)
            or not (OUT / f"line_{k:02d}.wav").exists()]
    if todo:
        import torch
        from qwen_tts import Qwen3TTSModel
        model = Qwen3TTSModel.from_pretrained(
            args.model, device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16, attn_implementation="sdpa")
        prompt = model.create_voice_clone_prompt(ref_audio=str(REF), ref_text=None, x_vector_only_mode=True)
        for k in todo:
            torch.manual_seed(args.seed + k)
            wavs, sr = model.generate_voice_clone(text=lines[k]["say"], language=script["language"],
                                                  voice_clone_prompt=prompt)
            wav = trim(np.asarray(wavs[0], dtype=np.float32), sr)
            sf.write(OUT / f"line_{k:02d}.wav", wav, sr)
            print(f"line {k:2d}: {len(wav) / sr:5.2f} s  {lines[k]['say']}")

    clips = [sf.read(OUT / f"line_{k:02d}.wav", dtype="float32") for k in range(len(lines))]
    sr = clips[0][1]
    timing, placed, t0, k = [], [], 0.0, 0
    for sc in script["scenes"]:
        t = sc["lead"]
        rows = []
        for j, ln in enumerate(sc["lines"]):
            d = len(clips[k][0]) / sr
            rows.append({"start": round(t, 3), "end": round(t + d, 3), "text": ln["text"]})
            placed.append((t0 + t, clips[k][0], ln["text"]))
            t += d + (GAP if j < len(sc["lines"]) - 1 else 0)
            k += 1
        dur = round((t + sc["tail"]) * FPS) / FPS
        timing.append({"id": sc["id"], "dur": dur, "lines": rows})
        t0 += dur

    track = np.zeros(int(t0 * sr) + 1, dtype=np.float32)
    srt = []
    for n, (s, wav, text) in enumerate(placed):
        a = int(s * sr)
        track[a:a + len(wav)] += wav[: len(track) - a]
        srt.append(f"{n + 1}\n{srt_time(s)} --> {srt_time(s + len(wav) / sr)}\n{text}\n")
    track *= 0.89 / max(1e-6, np.abs(track).max())
    sf.write(OUT / "narration.wav", track, sr)
    (OUT / "narration.srt").write_text("\n".join(srt))
    (HERE / "timing.js").write_text("// generated by voice.py from narration.json and the spoken clips\n"
                                   "const TIMING = " + json.dumps(timing, indent=1) + ";\n")
    for sc in timing:
        print(f"{sc['id']:<9} {sc['dur']:6.2f} s  " + "  ".join(f"[{r['start']:.2f}-{r['end']:.2f}]" for r in sc["lines"]))
    print(f"total {t0:.2f} s")


if __name__ == "__main__":
    main()


--- [158] ANSWER ---
TTS is still running; it has generated 7 of the 16 lines. Meanwhile I'm writing the film: the stage, both characters and the scene logic.

--- [168] ANSWER ---
The voice is done: 15 lines, 112 s in total, and timing.js is written. Next I'm writing the film source.

--- [169] EDIT:Write ---
[Write /home/hofmannp/animation_test/disclosure-channels/two-channels.html]
<!doctype html>
<meta charset="utf-8">
<title>Two Channels · financial statements and the MD&amp;A</title>
<style>body{margin:0;background:#1d1b19;color:#ddd;font:13px system-ui;display:grid;place-items:center;min-height:100vh}canvas{max-width:100vw;max-height:90vh}.bar{display:flex;gap:6px;align-items:center;padding:6px}</style>
<canvas id="c"></canvas>
<script src="core.js"></script>
<script src="studio.js"></script>
<script src="cels.js"></script>
<script src="fonts/caveat.js"></script>
<script src="timing.js"></script>
<script>
'use strict';
/* BRIEF
Subject: Brown, Hinson & Tucker (2024, CAR). Firms report through two channels: the
  rigid, GAAP-ruled, audited financial statements (Item 8) and the flexible MD&A
  narrative (Item 7). When the statements are inadequate (value that GAAP's boxes
  cannot hold), managers carry more through the MD&A: non-GAAP measures and
  forward-looking statements. The investor's picture ends as a mosaic from both.
Duration ~112 s, set by the narration (timing.js). 16:9, 1920x1080, 24 fps.
Look: ink contour on warm paper with flat, slightly misregistered print fills.
  Blue = financial statements channel; terracotta = MD&A channel; ochre = intangibles.
Cast: the manager (suit, tie; faces right) and the investor (ochre turtleneck, round
  glasses, magnifier; faces left). Whole-pose drawings rebuilt per exposed pose from
  authored landmarks; inbetweens only between compatible poses; on twos; holds are still.
Hardest action: the manager pushes an "ideas" blob into the square GAAP duct: it jams,
  he strains (two pushes), a few blue pieces squeeze through, he steps back and looks.
  Contacts: back foot planted through the push; front foot lifts on the breakdown.
Camera: locked or slow eased moves only. No shake, no idle boil, no flicker.
Beat sheet (see TIMING): title → channels → fits → stuck → question → mdna → numbers
  → methods → mosaic → end. Sound: soft procedural cues; narration mixed afterwards.
Sources: Caveat font (SIL OFL, fonts/OFL.txt). Everything else drawn in code.
*/
const INK='#2a2521', PAPER='#f2ebdc', BLUE='#2f6190', BLUE_L='#c9d8e6', RED='#c25a3c', RED_L='#f0d2c4',
  OCHRE='#d9a23e', OCHRE_L='#f1dcae', SLATE='#5f6c79', TROUSER='#3e4651', SKIN='#f0d3b8', GREY='#8b8276', CREAM='#fbf6ea';
const FONT = new FontFace('Caveat', `url(${CAVEAT_URL})`, {weight: '400 700'});
document.fonts.add(FONT); _photoLoads.push(FONT.load());
const SC = Object.fromEntries(TIMING.map(s => [s.id, s]));
const LN = (id, k) => SC[id].lines[k];                       // {start, end, text}
const at = (id, k, f) => { const l = LN(id, k); return l.start + (l.end - l.start) * f; };  // time at fraction f of a spoken line
const snap2 = t => Math.floor(t * 12 + 1e-6) / 12;          // character exposure: twos

// ---------------------------------------------------------------- small vector kit
const vadd = (a, b) => [a[0] + b[0], a[1] + b[1]], vsub = (a, b) => [a[0] - b[0], a[1] - b[1]], vmul = (a, k) => [a[0] * k, a[1] * k];
const vunit = a => { const l = Math.hypot(a[0], a[1]) || 1; return [a[0] / l, a[1] / l]; }, vperp = a => [-a[1], a[0]];
const vlerp = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t)];
const vrot = (p, r) => [p[0] * Math.cos(r) - p[1] * Math.sin(r), p[0] * Math.sin(r) + p[1] * Math.cos(r)];
function lerpDeep(a, b, t) { if (typeof a === 'number') return lerp(a, b, t); if (Array.isArray(a)) return a.map((v, i) => lerpDeep(v, b[i], t)); if (a && typeof a === 'object') { const o = {}; for (const k in a) o[k] = k in b ? lerpDeep(a[k], b[k], t) : a[k]; return o; } return t < .5 ? a : b; }

// ---------------------------------------------------------------- lettering and lines
function write(c, text, x, y, o = {}) {
  const {size = 44, color = INK, p = 1, align = 'left', weight = 600, rot = 0, alpha: al = 1} = o;
  const n = Math.round(text.length * clamp(p, 0, 1)); if (!n) return;
  c.save(); c.translate(x, y); c.rotate(rot); c.font = `${weight} ${size}px Caveat, ${HAND_FONT}`; c.textBaseline = 'alphabetic';
  const full = c.measureText(text).width, x0 = align === 'center' ? -full / 2 : align === 'right' ? -full : 0;
  c.textAlign = 'left'; c.fillStyle = color; c.globalAlpha *= al; c.fillText(text.slice(0, n), x0, 0); c.restore();
  return full;
}
const writeP = (tau, t0, text, cps = 26) => clamp((tau - t0) * cps / Math.max(1, text.length), 0, 1);   // write-on progress
function inkLine(c, pts, {w = 2.4, color = INK, seed = 1, p = 1, amp = 1.2, close = false, alpha: al = 1} = {}) {
  if (p <= 0) return; c.save(); c.strokeStyle = color; c.lineWidth = w; c.lineCap = 'round'; c.lineJoin = 'round'; c.globalAlpha *= al;
  if (p < 1) { const L = pathLength(smoothPts(pts, close), close) * 1.04 + 8; c.setLineDash([L * p, L * 2]); }
  wob(c, pts, amp, seed, close, {pressure: .4}); c.restore();
}
function printFill(c, path, color, {off = [-3, 2], al = 1} = {}) { c.save(); c.translate(off[0], off[1]); c.globalAlpha *= al; c.fillStyle = color; c.fill(path); c.restore(); }

// ---------------------------------------------------------------- characters
// A pose is a set of authored landmarks in local space (feet at 0,0; +x = facing).
// buildFigure turns a pose into a whole drawing: continuous contours, fills, face.
const MAN = {
  rest:    {hip:[0,-160], neck:[3,-300], head:[8,-345], tilt:0, eye:1, look:[1,0], brow:0, mouth:.25, open:0, hunch:0,
            armN:[[16,-224],[20,-168]], armF:[[-16,-226],[-18,-170]], legN:[[8,-80],[12,0]], legF:[[-8,-80],[-12,0]], fingers:0},
  hold:    {hip:[0,-160], neck:[4,-300], head:[10,-345], tilt:.04, eye:1, look:[1,.2], brow:.1, mouth:.3, open:0, hunch:0,
            armN:[[26,-232],[68,-244]], armF:[[18,-238],[60,-258]], legN:[[8,-80],[12,0]], legF:[[-8,-80],[-12,0]], fingers:0},
  step:    {hip:[16,-164], neck:[34,-302], head:[46,-345], tilt:.08, eye:1, look:[1,.1], brow:.2, mouth:.1, open:0, hunch:.2,
            armN:[[58,-240],[104,-252]], armF:[[50,-246],[96,-266]], legN:[[52,-104],[56,-30]], legF:[[-2,-80],[-12,0]], fingers:0},
  push1:   {hip:[30,-150], neck:[80,-282], head:[102,-322], tilt:.2, eye:1, look:[1,.1], brow:-.3, mouth:-.1, open:.1, hunch:.3,
            armN:[[122,-250],[166,-262]], armF:[[116,-258],[160,-274]], legN:[[84,-80],[95,0]], legF:[[6,-76],[-12,0]], fingers:0},
  push2:   {hip:[42,-140], neck:[96,-262], head:[122,-292], tilt:.38, eye:0, look:[1,0], brow:-1, mouth:-.5, open:.35, hunch:.6,
            armN:[[134,-236],[176,-258]], armF:[[128,-244],[170,-270]], legN:[[100,-86],[100,0]], legF:[[14,-70],[-12,0]], fingers:0},
  recoil:  {hip:[8,-160], neck:[2,-300], head:[2,-345], tilt:-.12, eye:1.2, look:[.4,-.4], brow:.9, mouth:-.35, open:.25, hunch:0,
            armN:[[30,-226],[58,-196]], armF:[[-10,-228],[20,-196]], legN:[[20,-80],[30,0]], legF:[[-6,-80],[-12,0]], fingers:1},
  look:    {hip:[2,-160], neck:[6,-300], head:[16,-342], tilt:.22, eye:1, look:[1,1], brow:.6, mouth:-.15, open:0, hunch:.1,
            armN:[[20,-224],[26,-168]], armF:[[-14,-226],[-14,-170]], legN:[[20,-80],[30,0]], legF:[[-6,-80],[-12,0]], fingers:0},
  think:   {hip:[2,-160], neck:[4,-300], head:[10,-346], tilt:-.1, eye:1, look:[.7,-1], brow:.5, mouth:-.1, open:0, hunch:0,
            armN:[[44,-240],[40,-302]], armF:[[20,-236],[52,-240]], legN:[[20,-80],[30,0]], legF:[[-6,-80],[-12,0]], fingers:0},
  write1:  {hip:[4,-160], neck:[22,-298], head:[40,-336], tilt:.32, eye:1, look:[1,1.2], brow:.2, mouth:.3, open:0, hunch:.3,
            armN:[[60,-214],[108,-196]], armF:[[40,-222],[84,-190]], legN:[[20,-80],[30,0]], legF:[[-6,-80],[-12,0]], fingers:0},
  write2:  {hip:[6,-160], neck:[25,-298], head:[44,-335], tilt:.3, eye:1, look:[1.2,1.1], brow:.25, mouth:.45, open:0, hunch:.3,
            armN:[[68,-212],[130,-192]], armF:[[40,-222],[84,-190]], legN:[[20,-80],[30,0]], legF:[[-6,-80],[-12,0]], fingers:0},
  pleased: {hip:[0,-160], neck:[4,-302], head:[12,-348], tilt:-.06, eye:1, look:[1,-.1], brow:.35, mouth:.9, open:.2, hunch:0,
            armN:[[22,-222],[40,-176]], armF:[[-14,-226],[-18,-170]], legN:[[14,-80],[20,0]], legF:[[-6,-80],[-12,0]], fingers:0},
};
const INV = {
  rest:    {hip:[0,-166], neck:[0,-312], head:[4,-358], tilt:0, eye:1, look:[1,0], brow:0, mouth:.1, open:0, hunch:0,
            armN:[[22,-238],[44,-190]], armF:[[-14,-236],[-16,-180]], legN:[[6,-84],[10,0]], legF:[[-8,-84],[-12,0]], fingers:0},
  lens:    {hip:[0,-166], neck:[6,-312], head:[16,-356], tilt:.1, eye:1, look:[1,-.2], brow:.2, mouth:0, open:0, hunch:.1,
            armN:[[48,-262],[84,-300]], armF:[[-14,-236],[-16,-180]], legN:[[6,-84],[10,0]], legF:[[-8,-84],[-12,0]], fingers:0},
  puzzled: {hip:[0,-166], neck:[-2,-312], head:[-4,-356], tilt:-.28, eye:1, look:[1,-.6], brow:.9, mouth:-.45, open:.15, hunch:0,
            armN:[[48,-262],[84,-296]], armF:[[-40,-282],[-20,-368]], legN:[[6,-84],[10,0]], legF:[[-8,-84],[-12,0]], fingers:1},
  pleased: {hip:[0,-166], neck:[2,-314], head:[8,-360], tilt:.05, eye:1, look:[1,0], brow:.4, mouth:.95, open:.25, hunch:0,
            armN:[[40,-250],[74,-262]], armF:[[-14,-236],[-16,-180]], legN:[[6,-84],[10,0]], legF:[[-8,-84],[-12,0]], fingers:0},
};
const HEAD = [[0,-44],[22,-40],[35,-24],[40,-2],[37,17],[26,35],[8,45],[-12,43],[-29,32],[-38,12],[-38,-12],[-27,-33],[-12,-42]];
const HAIR_M = [[-40,-2],[-42,-28],[-27,-48],[2,-54],[29,-46],[41,-26],[33,-30],[14,-37],[-6,-33],[-24,-22],[-32,-6]];
const HAIR_I = [[-40,6],[-43,-22],[-31,-44],[-8,-54],[18,-52],[36,-40],[40,-24],[26,-36],[4,-42],[-18,-36],[-30,-20],[-34,4]];
function limb(pts, ws) {                       // continuous sleeve / trouser contour from an authored centreline
  const [a, b, c] = pts, q = [a, vlerp(a, b, .5), b, vlerp(b, c, .5), c], w = [ws[0], lerp(ws[0], ws[1], .5), ws[1], lerp(ws[1], ws[2], .5), ws[2]];
  const n = q.map((p, i) => vperp(vunit(vsub(q[Math.min(4, i + 1)], q[Math.max(0, i - 1)]))));
  const L = q.map((p, i) => vadd(p, vmul(n[i], w[i]))), R = q.map((p, i) => vsub(p, vmul(n[i], w[i])));
  const bend = Math.sign((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]));
  return {L, R, fill: polyPath(smoothPts([...L, ...R.slice().reverse()], true, 3, .6)), bend, dir: vunit(vsub(c, b)), end: c, mid: b, n: n[2]};
}
function buildFigure(P, kind) {
  const strokes = [], fills = [], order = [];         // order: [{fills:[[path,color]], strokes:[...] }]
  let group;
  const begin = () => { group = {fills: [], strokes: []}; order.push(group); };
  const S_ = (id, points, width = 2.6, extra = {}) => group.strokes.push({id, points, width, opacity: 1, pressure: [[0, .25], [.15, .9], [.6, 1], [1, .3]], ...extra});
  const F_ = (path, color) => group.fills.push([path, color]);
  const down = vunit(vsub(P.hip, P.neck)), n = vperp(vmul(down, -1)), T = t => vlerp(P.hip, P.neck, t);
  const shN = vadd(vadd(P.neck, vmul(down, 24)), vmul(n, -4)), shF = vadd(vadd(P.neck, vmul(down, 20)), vmul(n, 8));
  const cloth = kind === 'man' ? SLATE : OCHRE, clothL = kind === 'man' ? '#7b8793' : '#e7bb62';
  const arm = (id, sh, [el, hd]) => {
    begin(); const A = limb([sh, el, hd], [13, 11, 9]);
    F_(A.fill, cloth); S_(id + '/a', A.L); S_(id + '/b', A.R);
    const cuffA = vadd(hd, vmul(A.n, 10)), cuffB = vsub(hd, vmul(A.n, 10));
    S_(id + '/cuff', [cuffA, vadd(vlerp(cuffA, cuffB, .5), vmul(A.dir, -3)), cuffB], 1.8);
    const crease = A.bend > 0 ? A.R : A.L;           // the concave side of the elbow gets a short crease
    S_(id + '/crease', [vlerp(crease[1], crease[2], .6), vadd(crease[2], vmul(A.n, A.bend > 0 ? 4 : -4)), vlerp(crease[2], crease[3], .35)], 1.3);
    const d = A.dir, nn = vperp(d), H = (x, y) => vadd(vadd(hd, vmul(d, x)), vmul(nn, y));
    const f = P.fingers || 0;
    const hand = [H(0, -8), H(12, -10), H(22 + f * 4, -7), H(26 + f * 6, 0), H(22 + f * 4, 7), H(10, 9), H(0, 8)];
    F_(polyPath(smoothPts(hand, true, 3, 1)), SKIN); S_(id + '/hand', [...hand, hand[0]], 2.1);
    S_(id + '/thumb', [H(8, -9), H(14, -17 - f * 3), H(20, -14 - f * 2)], 1.8);
  };
  const leg = (id, [kn, ft]) => {
    begin(); const hipP = vadd(P.hip, vmul(n, id === 'legN' ? 12 : -12)), A = limb([hipP, kn, vadd(ft, [0, -12])], [18, 15, 12]);
    F_(A.fill, kind === 'man' ? TROUSER : '#4c4a58'); S_(id + '/a', A.L, 2.4); S_(id + '/b', A.R, 2.4);
    const shoe = [[-12, -14], [12, -16], [30, -10], [38, -2], [36, 2], [-14, 2]].map(p => vadd(ft, p));
    F_(polyPath(smoothPts(shoe, true, 3, 1.2)), INK); S_(id + '/shoe', [...shoe, shoe[0]], 2);
  };
  arm('armF', shF, P.armF); leg('legF', P.legF); leg('legN', P.legN);
  // torso: one jacket contour, collar to hem to back; hunch rounds the back
  begin();
  const cF = vadd(P.neck, vmul(n, 22)), chF = vadd(T(.64), vmul(n, 36)), wF = vadd(T(.28), vmul(n, 35)), hemF = vadd(vadd(P.hip, vmul(down, 18)), vmul(n, 37));
  const cB = vadd(P.neck, vmul(n, -21)), chB = vadd(T(.64), vmul(n, -34 - P.hunch * 12)), wB = vadd(T(.28), vmul(n, -33)), hemB = vadd(vadd(P.hip, vmul(down, 18)), vmul(n, -35));
  const hemM = vadd(P.hip, vmul(down, 22));
  const body = [cF, chF, wF, hemF, hemM, hemB, wB, chB, cB];
  F_(polyPath(smoothPts(body, true, 3, 1.1)), cloth);
  S_('torso/front', [cF, chF, wF, hemF], 2.8); S_('torso/hem', [hemF, hemM, hemB], 2.4); S_('torso/back', [hemB, wB, chB, cB], 2.8);
  if (kind === 'man') {
    const v = [vadd(P.neck, vmul(n, 14)), vadd(T(.62), vmul(n, 15)), vadd(P.neck, vmul(n, -6))];
    F_(polyPath(v), CREAM);
    const tie = [vadd(vadd(P.neck, vmul(n, 6)), vmul(down, 6)), vadd(T(.84), vmul(n, 12)), vadd(T(.56), vmul(n, 14)), vadd(T(.62), vmul(n, 6)), vadd(T(.84), vmul(n, 3))];
    F_(polyPath(tie), RED); S_('tie', [...tie, tie[0]], 1.6);
    S_('lapel/f', [vadd(P.neck, vmul(n, 18)), vadd(T(.62), vmul(n, 17)), vadd(T(.5), vmul(n, 22))], 1.8);
    S_('lapel/b', [vadd(P.neck, vmul(n, -8)), vadd(T(.7), vmul(n, 2)), vadd(T(.5), vmul(n, 8))], 1.5);
    S_('button', [vadd(T(.38), vmul(n, 22)), vadd(T(.36), vmul(n, 25)), vadd(T(.34), vmul(n, 22))], 2.2);
  } else {
    const col = [vadd(P.neck, vmul(n, 20)), vadd(vadd(P.neck, vmul(down, -14)), vmul(n, 16)), vadd(vadd(P.neck, vmul(down, -14)), vmul(n, -16)), vadd(P.neck, vmul(n, -20))];
    F_(polyPath(smoothPts(col, true, 3, 1.2)), clothL); S_('collar', [...col, col[0]], 2);
    for (let k = 0; k < 5; k++) { const u = (k + .5) / 5; const a = vlerp(hemF, hemB, u); S_('rib/' + k, [vadd(a, vmul(down, -12)), a], 1.2, {opacity: .7}); }
  }
  // head
  begin();
  const R = P.tilt, Hp = p => vadd(P.head, vrot(p, R));
  const head = HEAD.map(Hp);
  F_(polyPath(smoothPts(head, true, 3, 2)), SKIN); S_('head', [...head, head[0]], 2.6, {close: false});
  S_('ear', [[-10, -8], [-19, -12], [-21, 3], [-12, 9]].map(Hp), 1.8);
  const hair = (kind === 'man' ? HAIR_M : HAIR_I).map(Hp);
  F_(polyPath(smoothPts(hair, true, 3, 1.2)), kind === 'man' ? '#3b3129' : '#9d968c'); S_('hair', [...hair, hair[0]], 1.8);
  S_('nose', [[35, -8], [45, 7], [36, 12]].map(Hp), 2, {corner: 1});
  const eyeY = -2, eyes = [[12, eyeY], [30, eyeY - 1]];
  const b = P.brow;
  eyes.forEach(([ex, ey], k) => {
    const inner = k === 0 ? 5 : -5, y0 = ey - 15 - b * 5;
    S_('brow/' + k, [[ex - 6, y0 + (k === 0 ? b * 3 : -b * 3) * 0], [ex, y0 - 1], [ex + 6, y0]].map(([x, y], j) => Hp([x, y + (j === (k === 0 ? 2 : 0) ? -b * 3 : 0)])), 2);
  });
  const m = P.mouth, mo = P.open;
  const mouth = [[13, 23], [19, 24 + m * 6], [27, 24 + m * 6], [33, 21]].map(Hp);
  S_('mouth', mouth, 2);
  if (kind === 'inv') {                                  // round glasses
    eyes.forEach(([ex, ey], k) => S_('glass/' + k, ellPts(ex, ey, 9.5, 9, 0, 12).map(Hp).concat([Hp([ex + 9.5, ey])]), 1.6));
    S_('glass/bridge', [[21, eyeY - 1], [23, eyeY - 3], [25, eyeY - 1]].map(Hp), 1.4);
  }
  const face = {eyes: eyes.map(e => Hp(e)), eye: P.eye, look: P.look, tilt: R, mouth, open: mo, m, Hp};
  // near arm last, in front of the body
  arm('armN', shN, P.armN);
  return {order, face};
}
const _figCache = new Map();
function figure(kind, key, P) {
  const k = kind + '|' + key;
  if (!_figCache.has(k)) {
    if (_figCache.size > 400) _figCache.delete(_figCache.keys().next().value);
    const f = buildFigure(P, kind);
    f.order = f.order.map((g, i) => ({fills: g.fills, cel: compileCel({strokes: g.strokes}, {id: kind + '/g' + i})}));
    _figCache.set(k, f);
  }
  return _figCache.get(k);
}
function drawFigure(c, f) {
  for (const g of f.order) { for (const [p, col] of g.fills) printFill(c, p, col); drawCel(c, g.cel, {material: 'ink', color: INK}); }
  const {eyes, eye, look, Hp, tilt, mouth, open} = f.face;
  c.save(); c.fillStyle = INK;
  for (const [x, y] of eyes) {
    if (eye > .35) { c.beginPath(); c.ellipse(x + look[0] * 1.6, y + look[1] * 1.6, 3.4, 4.6 * Math.min(eye, 1.2), tilt, 0, TAU); c.fill(); }
    else { c.strokeStyle = INK; c.lineWidth = 2.2; c.lineCap = 'round'; c.beginPath(); c.moveTo(x - 6, y + 1); c.quadraticCurveTo(x, y - 3, x + 6, y + 1); c.stroke(); }
  }
  if (open > .05) { const lo = mouth.map((p, i) => i === 0 || i === 3 ? p : vadd(p, vrot([0, open * 12], tilt))); c.fillStyle = '#6e2a22'; c.fill(polyPath(smoothPts([...mouth, ...lo.slice(1, 3).reverse()], true, 2, 2))); }
  c.restore();
}
// Pose timelines: [[time, poseName], ...]; the same pose twice is a hold. Sampled on twos.
function poseAt(set, keys, t, ease = easeInOutSine) {
  t = snap2(t);
  if (t <= keys[0][0]) return {key: keys[0][1], P: set[keys[0][1]]};
  for (let k = 0; k < keys.length - 1; k++) {
    const [t0, a] = keys[k], [t1, b, e] = keys[k + 1];
    if (t < t1) { if (a === b) return {key: a, P: set[a]}; const u = Math.round((e || ease)(clamp((t - t0) / (t1 - t0), 0, 1)) * 48) / 48;
      return {key: `${a}>${b}@${u}`, P: lerpDeep(set[a], set[b], u)}; }
  }
  const last = keys[keys.length - 1][1]; return {key: last, P: set[last]};
}
function actor(c, kind, keys, t, x, y = 900, {dir = kind === 'man' ? 1 : -1, scale = 1, ease} = {}) {
  const set = kind === 'man' ? MAN : INV, {key, P} = poseAt(set, keys, t, ease), f = figure(kind, key, P);
  c.save(); c.translate(x, y); c.scale(dir * scale, scale); drawFigure(c, f); c.restore();
  return {P, local: p => [x + dir * scale * p[0], y + scale * p[1]]};
}
// ---------------------------------------------------------------- stage
const DUCT = {x0: 500, x1: 1345, y: 640, h: 130};
const RIB = motionPath([[610, 800], [700, 812], [820, 784], [960, 818], [1100, 790], [1240, 812], [1340, 790], [1400, 752], [1430, 730]]);
const BOARD = {x: 1400, y: 550, cols: 4, rows: 3, s: 60};
const slot = k => [BOARD.x + (k % 4) * 60 + 30, BOARD.y + Math.floor(k / 4) * 60 + 30];
function ductMouthIn(c, p = 1) {   // the square opening, drawn over whatever enters it
  const {x0, y, h} = DUCT;
  c.save(); c.globalAlpha *= clamp(p * 1.5, 0, 1);
  const outer = rectPath(x0 - 14, y - h / 2 - 14, 40, h + 28);
  printFill(c, outer, BLUE, {off: [-2, 2]});
  inkLine(c, [[x0 - 14, y - h / 2 - 14], [x0 + 26, y - h / 2 - 14], [x0 + 26, y + h / 2 + 14], [x0 - 14, y + h / 2 + 14]], {close: true, seed: 81, w: 2.6});
  c.restore();
}
function duct(c, p = 1, {label = 1, stamp = 0, audit = 0} = {}) {
  const {x0, x1, y, h} = DUCT, top = y - h / 2, bot = y + h / 2;
  if (p <= 0) return;
  const fa = clamp((p - .4) / .6, 0, 1);
  c.save(); c.globalAlpha *= fa;
  printFill(c, rectPath(x0, top, x1 - x0, h), BLUE_L, {off: [-3, 3]});
  c.strokeStyle = alpha(BLUE, .35); c.lineWidth = 1.2; c.beginPath();
  for (let yy = top + 16; yy < bot - 6; yy += 16) { c.moveTo(x0 + 30, yy); c.lineTo(x1 - 8, yy); }       // ledger ruling
  for (const xx of [x0 + 140, x0 + 420, x0 + 700]) { c.moveTo(xx, top + 4); c.lineTo(xx, bot - 4); }
  c.stroke();
  c.fillStyle = BLUE; for (let xx = x0 + 60; xx < x1; xx += 90) for (const yy of [top + 8, bot - 8]) { c.beginPath(); c.arc(xx, yy, 2.6, 0, TAU); c.fill(); }
  c.restore();
  inkLine(c, [[x0, top], [x1, top]], {p: clamp(p * 1.4, 0, 1), seed: 11, w: 3, color: INK});
  inkLine(c, [[x0, bot], [x1, bot]], {p: clamp(p * 1.4 - .15, 0, 1), seed: 12, w: 3, color: INK});
  inkLine(c, [[x1, top], [x1, bot]], {p: clamp(p * 2 - 1, 0, 1), seed: 13, w: 2.6});
  if (label) {
    write(c, 'Item 8 · Financial statements', x0 + 10, top - 26, {size: 40, color: BLUE, p: label, weight: 700});
  }
  if (stamp > 0) {                                     // the GAAP stamp lands: scale from 1.35 to 1 on ones
    const s = 1 + .35 * (1 - easeOutQuint(clamp(stamp, 0, 1))), a = clamp(stamp * 3, 0, 1);
    c.save(); c.translate(x0 + 480, y); c.rotate(-.08); c.scale(s, s); c.globalAlpha *= a;
    c.strokeStyle = RED; c.lineWidth = 4; c.strokeRect(-92, -34, 184, 68); c.lineWidth = 1.5; c.strokeRect(-84, -26, 168, 52);
    write(c, 'GAAP rules', 0, 14, {size: 42, color: RED, align: 'center', weight: 700}); c.restore();
  }
  if (audit > 0) {
    c.save(); c.translate(x1 - 150, top - 44);
    inkLine(c, [[-8, 0], [4, 12], [26, -16]], {p: clamp(audit * 1.6, 0, 1), seed: 16, w: 4.5, color: BLUE, amp: .6});
    write(c, 'audited', 34, 8, {size: 34, color: BLUE, p: clamp(audit * 1.6 - .5, 0, 1)}); c.restore();
  }
}
function ribbon(c, p = 1, {label = 1} = {}) {
  if (p <= 0) return;
  const n = 90, L = [], R = [], w = 30;
  for (let i = 0; i <= n; i++) { const u = i / n * p, a = RIB.at(u), nn = vperp(a.tangent), ww = w * (1 - .35 * u); L.push(vadd(a.p, vmul(nn, ww))); R.push(vsub(a.p, vmul(nn, ww))); }
  const band = polyPath([...L, ...R.slice().reverse()]);
  printFill(c, band, RED_L, {off: [-3, 3]}); printFill(c, band, CREAM, {off: [0, 0], al: .55});
  c.save(); c.clip(band); c.strokeStyle = alpha(RED, .45); c.lineWidth = 2; c.lineCap = 'round';
  for (let i = 0; i < 26; i++) { const u0 = i / 26 + .006, u1 = u0 + .022 + hash(i, 4) * .012; if (u1 > p) break;
    for (const off of [-10, 2, 13]) { if (hash(i * 3 + off, 5) < .2) continue; const a = RIB.at(u0), b = RIB.at(u1), na = vperp(a.tangent), nb = vperp(b.tangent);
      c.beginPath(); c.moveTo(...vadd(a.p, vmul(na, off * (1 - .35 * u0)))); c.lineTo(...vadd(b.p, vmul(nb, off * (1 - .35 * u1)))); c.stroke(); } }
  c.restore();
  c.save(); c.strokeStyle = RED; c.lineWidth = 2.6; c.lineCap = 'round';
  c.beginPath(); L.forEach((q, i) => i ? c.lineTo(...q) : c.moveTo(...q)); c.stroke();
  c.beginPath(); R.forEach((q, i) => i ? c.lineTo(...q) : c.moveTo(...q)); c.stroke(); c.restore();
  if (label) write(c, "Item 7 · MD&A: management's narrative", 690, 890, {size: 40, color: RED, p: label, weight: 700});
}
function lectern(c) {
  const top = 792;
  const desk = polyPath([[540, top], [640, top - 10], [648, top + 6], [548, top + 16]]);
  printFill(c, desk, '#b88a5a'); inkLine(c, [[540, top], [640, top - 10], [648, top + 6], [548, top + 16]], {close: true, seed: 31, w: 2.4});
  inkLine(c, [[560, top + 14], [566, 900]], {seed: 32, w: 3}); inkLine(c, [[628, top + 4], [622, 900]], {seed: 33, w: 3});
  inkLine(c, [[556, 900], [640, 900]], {seed: 34, w: 2.2});
}
function ground(c) { c.save(); c.globalAlpha = .55; inkLine(c, [[40, 904], [520, 902]], {seed: 41, w: 1.8, color: GREY}); inkLine(c, [[1560, 902], [1880, 904]], {seed: 42, w: 1.8, color: GREY}); c.restore(); }
function building(c, kind, p = 1) {
  if (p <= 0) return;
  c.save(); c.globalAlpha *= clamp(p, 0, 1);
  if (kind === 'office') {
    const b = [[60, 900], [60, 470], [200, 440], [200, 900]];
    printFill(c, polyPath(b), '#e3dccb'); inkLine(c, b, {seed: 51, w: 2.2, color: GREY});
    for (let r = 0; r < 6; r++) for (let q = 0; q < 2; q++) c.strokeStyle = GREY, c.lineWidth = 1.6, c.strokeRect(86 + q * 58, 500 + r * 62, 32, 36);
    write(c, 'the firm', 130, 436, {size: 34, color: GREY, align: 'center'});
  } else if (kind === 'factory') {
    const b = [[40, 900], [40, 640], [95, 590], [95, 640], [150, 590], [150, 640], [205, 590], [205, 900]];
    printFill(c, polyPath(b), '#e0cfc0'); inkLine(c, b, {seed: 52, w: 2.4, color: INK, amp: .8});
    inkLine(c, [[165, 590], [165, 470], [190, 470], [190, 590]], {seed: 53, w: 2.4});
    c.fillStyle = alpha(GREY, .35); for (const [x, y, r] of [[182, 440, 18], [200, 405, 24], [226, 368, 28]]) { c.beginPath(); c.arc(x, y, r, 0, TAU); c.fill(); }
    for (let q = 0; q < 3; q++) c.strokeStyle = INK, c.lineWidth = 1.6, c.strokeRect(58 + q * 50, 700, 30, 40);
    write(c, 'factory, machines, inventory', 30, 960, {size: 32, color: GREY});
  } else {
    const b = [[50, 900], [50, 520], [210, 520], [210, 900]];
    printFill(c, polyPath(b), OCHRE_L); inkLine(c, [...b, b[0]], {seed: 54, w: 2.4});
    bulb(c, 130, 610, 1.1);
    for (let r = 0; r < 2; r++) for (let q = 0; q < 3; q++) c.strokeStyle = INK, c.lineWidth = 1.6, c.strokeRect(70 + q * 45, 720 + r * 70, 28, 44);
    write(c, 'ideas, software, brands', 30, 960, {size: 32, color: GREY});
  }
  c.restore();
}
function bulb(c, x, y, s = 1, color = OCHRE) {
  c.save(); c.translate(x, y); c.scale(s, s);
  const g = [[-20, 8], [-30, -12], [-24, -36], [0, -48], [24, -36], [30, -12], [20, 8], [14, 22], [-14, 22]];
  printFill(c, polyPath(smoothPts(g, true, 3, 1)), color, {off: [-2, 2]}); inkLine(c, [...g, g[0]], {seed: 61, w: 2.2, amp: .5});
  inkLine(c, [[-12, 30], [12, 30]], {seed: 62, w: 2.2, amp: .3}); inkLine(c, [[-9, 37], [9, 37]], {seed: 63, w: 2.2, amp: .3});
  inkLine(c, [[-7, 20], [-4, -8], [0, 0], [4, -8], [7, 20]], {seed: 64, w: 1.6, amp: .3}); c.restore();
}
function gear(c, x, y, r, color = INK) {
  c.save(); c.translate(x, y); const pts = []; for (let k = 0; k < 24; k++) { const a = k / 24 * TAU, rr = (k % 3 === 0) ? r * 1.25 : r; pts.push([Math.cos(a) * rr, Math.sin(a) * rr]); }
  c.strokeStyle = color; c.lineWidth = 2.2; c.lineJoin = 'round'; c.stroke(polyPath(pts)); c.beginPath(); c.arc(0, 0, r * .35, 0, TAU); c.stroke(); c.restore();
}
// the investor's picture of the firm: a 4x3 board; each tile shows part of the picture
function boardPicture(c, pic) {
  const {x, y} = BOARD;
  if (pic === 'factory') {
    const b = [[x + 20, y + 170], [x + 20, y + 90], [x + 70, y + 55], [x + 70, y + 90], [x + 120, y + 55], [x + 120, y + 90], [x + 170, y + 55], [x + 170, y + 170]];
    c.fillStyle = alpha(INK, .75); c.fill(polyPath(b)); c.fillRect(x + 185, y + 25, 22, 145);
  } else {
    bulb(c, x + 120, y + 105, 1.55, alpha(INK, .6));
    c.strokeStyle = alpha(INK, .75); c.lineWidth = 5; c.lineCap = 'round'; c.beginPath(); c.moveTo(x + 14, y + 165); c.lineTo(x + 70, y + 130); c.lineTo(x + 110, y + 148); c.lineTo(x + 226, y + 30); c.stroke();
  }
}
function board(c, tiles, tau, {pic = 'factory', label = 1, gaps = 0} = {}) {
  const {x, y, s} = BOARD, w = s * 4, h = s * 3;
  printFill(c, rectPath(x - 12, y - 12, w + 24, h + 24), '#e7dfcf');
  inkLine(c, [[x - 12, y - 12], [x + w + 12, y - 12], [x + w + 12, y + h + 12], [x - 12, y + h + 12]], {close: true, seed: 71, w: 3});
  c.save(); c.strokeStyle = alpha(GREY, .5); c.lineWidth = 1.2; c.setLineDash([5, 6]);
  for (let k = 0; k < 12; k++) { const [cx, cy] = slot(k); c.strokeRect(cx - 28, cy - 28, 56, 56); } c.restore();
  if (gaps > 0) { c.save(); c.globalAlpha *= clamp(gaps, 0, 1); c.strokeStyle = RED; c.lineWidth = 3; c.setLineDash([8, 7]);
    for (let k = 0; k < 12; k++) if (!tiles.some(t => t.k === k && tau >= t.t)) { const [cx, cy] = slot(k); c.strokeRect(cx - 26, cy - 26, 52, 52); } c.restore(); }
  for (const t of tiles) {
    const u = clamp((tau - t.t + .55) / .55, 0, 1); if (u <= 0) continue;
    const [sx, sy] = slot(t.k), from = t.from || (t.c === 'b' ? [DUCT.x1 + 10, DUCT.y] : [1430, 740]);
    const e = easeInOutSine(u), px = lerp(from[0], sx, e), py = lerp(from[1], sy, e) - Math.sin(Math.PI * e) * 60;
    c.save(); c.translate(px, py); const sc = u < 1 ? .6 + .4 * e : 1; c.scale(sc, sc);
    const tp = rectPath(-28, -28, 56, 56);
    c.fillStyle = t.c === 'b' ? BLUE_L : RED_L; c.fill(tp);
    if (u >= 1) { c.save(); c.clip(tp); c.translate(-sx, -sy); boardPicture(c, pic); c.restore(); }
    c.strokeStyle = t.c === 'b' ? BLUE : RED; c.lineWidth = 3; c.stroke(tp); c.restore();
  }
  if (label) write(c, "the investor's picture", x + w / 2, y + h + 56, {size: 34, color: GREY, align: 'center', p: label});
}
function crate(c, x, y, s = 1) {
  c.save(); c.translate(x, y); c.scale(s, s);
  const b = rectPath(-46, -46, 92, 92); printFill(c, b, BLUE_L); c.fillStyle = alpha(BLUE, .18); c.fill(b);
  inkLine(c, [[-46, -46], [46, -46], [46, 46], [-46, 46]], {close: true, seed: 91, w: 2.8, amp: .6});
  inkLine(c, [[-46, -46], [46, 46]], {seed: 92, w: 1.6, amp: .5, color: BLUE}); inkLine(c, [[46, -46], [-46, 46]], {seed: 93, w: 1.6, amp: .5, color: BLUE});
  c.restore();
}
// the "ideas" blob: soft, too big for the square mouth; squash k flattens it against the mouth plane
const BLOB = blob(0, 0, 105, 92, 7, {amp: .09, n: 40});
function ideaBlob(c, x, y, {k = 0, flatX = Infinity} = {}) {
  const sx = 1 - .18 * k, sy = 1 + .16 * k;
  const pts = BLOB.map(([px, py]) => { let X = x + px * sx, Y = y + py * sy; if (X > flatX) { const over = X - flatX; X = flatX + over * .08; Y += Math.sign(py) * over * .25; } return [X, Y]; });
  const path = polyPath(smoothPts(pts, true, 3, 2));
  printFill(c, path, OCHRE_L); c.save(); c.clip(path);
  c.globalAlpha = .9; bulb(c, x - 30 * sx, y - 8, .8); gear(c, x + 40 * sx, y - 34 * sy, 17); gear(c, x + 60 * sx, y + 2, 11);
  write(c, '</>', x + 18 * sx, y + 58 * sy, {size: 40, color: INK, align: 'center', weight: 700});
  write(c, '™', x - 62 * sx, y + 46 * sy, {size: 44, color: RED, align: 'center', weight: 700});
  c.restore();
  inkLine(c, [...pts, pts[0]], {seed: 95, w: 3, amp: .4});
}
// inset chart: price against what earnings + book value predict
function inset(c, tau, t0, {spread = 0, show = 1, label = '', formula = 0} = {}) {
  if (show <= 0) return;
  const x = 1300, y = 110, w = 420, h = 280, a = clamp(show, 0, 1);
  c.save(); c.globalAlpha *= a;
  printFill(c, rectPath(x, y, w, h), '#f8f2e4', {off: [0, 0]});
  c.strokeStyle = alpha(GREY, .6); c.lineWidth = 1.4; c.strokeRect(x, y, w, h);
  inkLine(c, [[x + 50, y + 30], [x + 50, y + h - 50], [x + w - 24, y + h - 50]], {seed: 101, w: 2.2});
  write(c, 'price', x + 16, y + 40, {size: 26, color: GREY}); write(c, 'earnings + book value', x + w - 24, y + h - 16, {size: 26, color: GREY, align: 'right'});
  inkLine(c, [[x + 70, y + h - 70], [x + w - 40, y + 50]], {seed: 102, w: 2, color: BLUE, alpha: .8, p: clamp((tau - t0) / .8, 0, 1)});
  for (let k = 0; k < 16; k++) {
    const u = (k + .5) / 16, px = lerp(x + 75, x + w - 45, u), py0 = lerp(y + h - 72, y + 52, u);
    const off = (hash(k, 3) - .5) * 2 * (6 + 120 * spread) ;
    const show = clamp((tau - t0 - .2 - k * .05) / .2, 0, 1); if (show <= 0) continue;
    c.fillStyle = INK; c.beginPath(); c.arc(px, clamp(py0 + off, y + 34, y + h - 58), 5.2 * show, 0, TAU); c.fill();
  }
  if (label) write(c, label, x + w - 36, y + 64 + (spread ? 170 : 0), {size: 38, color: spread ? RED : BLUE, align: 'right', weight: 700, p: clamp((tau - t0 - 1.1) * 2, 0, 1)});
  c.restore();
  if (formula > 0) {
    c.save(); c.translate(x + 10, y + h + 64);
    write(c, 'inadequacy = 1 − R²', 0, 0, {size: 50, color: RED, weight: 700, p: formula});
    c.restore();
  }
}
function bubble(c, x, y, w, h, fromX, fromY, a = 1, draw) {
  if (a <= 0) return; c.save(); c.globalAlpha *= clamp(a, 0, 1);
  const pts = blob(x, y, w / 2, h / 2, 17, {amp: .06, n: 36}), path = polyPath(smoothPts(pts, true, 3, 2));
  c.fillStyle = CREAM; c.fill(path); inkLine(c, [...pts, pts[0]], {seed: 111, w: 2.4, amp: .3});
  for (let k = 0; k < 3; k++) { const u = (k + 1) / 4, r = 5 + k * 3; c.beginPath(); c.arc(lerp(fromX, x - w * .15, u), lerp(fromY, y + h / 2, u), r, 0, TAU); c.fillStyle = CREAM; c.fill(); c.strokeStyle = INK; c.lineWidth = 2; c.stroke(); }
  if (draw) draw(c); c.restore();
}
function card(c, text, x, y, rot, color, {w = 190, h = 52, arrow = false} = {}) {
  c.save(); c.translate(x, y); c.rotate(rot);
  const r = rectPath(-w / 2, -h / 2, w, h); c.fillStyle = CREAM; c.fill(r); c.strokeStyle = color; c.lineWidth = 3; c.stroke(r);
  if (arrow) { c.strokeStyle = color; c.lineWidth = 3; c.beginPath(); c.moveTo(w / 2 - 36, 0); c.lineTo(w / 2 - 12, 0); c.moveTo(w / 2 - 20, -8); c.lineTo(w / 2 - 12, 0); c.lineTo(w / 2 - 20, 8); c.stroke(); }
  write(c, text, arrow ? -14 : 0, 10, {size: 30, color: INK, align: 'center', weight: 700}); c.restore();
}
function pen(c, x, y, rot = -.9) {
  c.save(); c.translate(x, y); c.rotate(rot);
  const b = [[0, 0], [10, -6], [70, -6], [74, 0], [70, 6], [10, 6]]; printFill(c, polyPath(b), RED, {off: [0, 0]}); inkLine(c, [...b, b[0]], {seed: 121, w: 2, amp: .2});
  c.restore();
}
function caption(c, text, tau, t0, {x = 90, y = 110, size = 56, color = INK} = {}) { write(c, text, x, y, {size, color, weight: 700, p: writeP(tau, t0, text, 30)}); }

// ---------------------------------------------------------------- scenes
function sceneTitle(c, tau) {
  paper(c, PAPER, null, 5);
  const l = LN('title', 0);
  const p = clamp((tau - .2) / 1.1, 0, 1);
  // the annual report, drawn once, then its two sections are tabbed
  const bx = CX - 170, by = 330, bw = 340, bh = 440;
  c.save(); c.globalAlpha = easeOut(p);
  printFill(c, rectPath(bx + 16, by + 12, bw, bh), '#ddd3bf'); printFill(c, rectPath(bx, by, bw, bh), '#f9f4e8', {off: [0, 0]});
  c.restore();
  inkLine(c, [[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]], {close: true, p, seed: 3, w: 3});
  write(c, 'Annual report', CX, by + 80, {size: 54, align: 'center', weight: 700, p: writeP(tau, .6, 'Annual report')});
  write(c, 'Form 10-K', CX, by + 124, {size: 34, align: 'center', color: GREY, p: writeP(tau, 1.0, 'Form 10-K')});
  c.save(); c.strokeStyle = alpha(GREY, .5); c.lineWidth = 2; for (let k = 0; k < 7; k++) { const q = clamp((tau - 1.2 - k * .08) / .3, 0, 1); if (q <= 0) continue; c.beginPath(); c.moveTo(bx + 40, by + 170 + k * 22); c.lineTo(bx + 40 + (bw - 90 - (k % 3) * 30) * q, by + 170 + k * 22); c.stroke(); } c.restore();
  const tab = (t0, color, light, label, y, side) => {
    const u = easeOutBack(clamp((tau - t0) / .5, 0, 1)); if (u <= 0) return;
    const x = side > 0 ? bx + bw : bx, len = 250 * u;
    c.save(); const r = side > 0 ? rectPath(x - 8, y - 32, len + 8, 64) : rectPath(x - len, y - 32, len + 8, 64);
    printFill(c, r, light); c.strokeStyle = color; c.lineWidth = 3; c.stroke(r);
    if (u > .9) write(c, label, side > 0 ? x + 20 : x - 20, y + 12, {size: 36, color, weight: 700, align: side > 0 ? 'left' : 'right'});
    c.restore();
  };
  tab(at('title', 0, .72), RED, RED_L, 'Item 7 · MD&A', by + 300, -1);
  tab(at('title', 0, .82), BLUE, BLUE_L, 'Item 8 · Financial statements', by + 380, 1);
  write(c, 'Two channels', CX, 190, {size: 104, align: 'center', weight: 700, p: writeP(tau, .3, 'Two channels', 16)});
  write(c, 'Financial statement adequacy and firms’ MD&A disclosures', CX, 262, {size: 40, align: 'center', color: GREY, p: writeP(tau, 1.3, 'Financial statement adequacy and firms’ MD&A disclosures', 40)});
  write(c, 'Brown, Hinson & Tucker · Contemporary Accounting Research (2024)', CX, 890, {size: 34, align: 'center', color: GREY, p: writeP(tau, 2.2, 'Brown, Hinson & Tucker · Contemporary Accounting Research (2024)', 45)});
}
function stageBase(c, {building: bk = 'office', ductP = 1, ribP = 1, labels = 1, stamp = 1, audit = 1} = {}) {
  paper(c, PAPER, null, 5);
  building(c, bk); ground(c);
  duct(c, ductP, {label: labels, stamp, audit}); lectern(c); ribbon(c, ribP, {label: labels});
}
function sceneChannels(c, tau) {
  const l0 = LN('channels', 0), l1 = LN('channels', 1);
  const ductP = clamp((tau - l0.start - .1) / 1.6, 0, 1), ribP = easeInOutSine(clamp((tau - l1.start - .2) / 2.4, 0, 1));
  paper(c, PAPER, null, 5); building(c, 'office'); ground(c);
  duct(c, ductP, {label: writeP(tau, l0.start + .6, 'Item 8 · Financial statements'), stamp: clamp((tau - at('channels', 0, .5)) / .35, 0, 1), audit: clamp((tau - at('channels', 0, .82)) / .6, 0, 1)});
  ductMouthIn(c, ductP);
  lectern(c); ribbon(c, ribP, {label: writeP(tau, l1.start + 1.2, "Item 7 · MD&A: management's narrative")});
  board(c, [], tau, {label: writeP(tau, 1, "the investor's picture")});
  actor(c, 'man', [[0, 'rest'], [l1.start + .2, 'rest'], [l1.start + .9, 'look'], [l1.end - .4, 'look'], [l1.end + .3, 'rest']], tau, 250);
  actor(c, 'inv', [[0, 'rest'], [l0.start + 1.8, 'rest'], [l0.start + 2.6, 'lens'], [l1.end, 'lens']], tau, 1770);
  write(c, 'strict: rules, audit', 780, 470, {size: 38, color: BLUE, p: writeP(tau, at('channels', 0, .6), 'strict: rules, audit'), rot: -.02});
  write(c, 'flexible: words, context, outlook', 760, 990, {size: 38, color: RED, p: writeP(tau, l1.start + 3.2, 'flexible: words, context, outlook')});
}
// push-the-object action shared by "fits" and "stuck"
function sceneFits(c, tau) {
  const l = LN('fits', 0);
  const tStep = .5, tPush = 1.3, tRel = 2.1;
  stageBase(c, {building: 'factory'});
  const tiles = Array.from({length: 12}, (_, k) => ({k, c: 'b', t: 3.0 + k * .12}));
  board(c, tiles, tau, {pic: 'factory'});
  const m = actor(c, 'man', [[0, 'hold'], [tStep, 'hold'], [tStep + .5, 'step'], [tPush, 'push1'], [tRel + .2, 'push1'], [tRel + .9, 'rest'], [5.2, 'rest'], [5.8, 'pleased']], tau, 250);
  // the crate rides the near hand until release, then slides into the duct at constant speed
  const hand = m.local(m.P.armN[1]);
  const tq = snap2(tau);
  let cx = hand[0] + 50, cy = hand[1] + 6;
  if (tq > tRel) { const u = clamp((tq - tRel) / 1.0, 0, 1); cx = lerp(MAN.push1.armN[1][0] + 250 + 50, DUCT.x0 + 160, u); cy = lerp(MAN.push1.armN[1][1] + 900 + 6, DUCT.y, clamp(u * 3, 0, 1)); }
  c.save(); c.beginPath(); c.rect(0, 0, DUCT.x0 + 26, H); c.clip(); crate(c, cx, cy); c.restore();
  ductMouthIn(c);
  actor(c, 'inv', [[0, 'lens'], [3.8, 'lens'], [4.4, 'pleased']], tau, 1770);
  inset(c, tau, l.start + 2.8, {spread: 0, label: 'R² high'});
  caption(c, 'Old economy: the statements capture the value', tau, l.start + .2);
}
function sceneStuck(c, tau) {
  const l0 = LN('stuck', 0), l1 = LN('stuck', 1);
  const t1 = .9, tC = 2.0, tS = 2.6;
  stageBase(c, {building: 'lab'});
  const tiles = [{k: 4, c: 'b', t: 3.6}, {k: 1, c: 'b', t: 4.3}, {k: 9, c: 'b', t: 5.0}];
  board(c, tiles, tau, {pic: 'bulb', gaps: clamp((tau - l1.start - 1.8) / .4, 0, 1)});
  const keys = [[0, 'hold'], [t1 - .3, 'hold'], [t1 + .2, 'step'], [tC, 'push1'], [tS, 'push2'], [3.2, 'push1'], [3.7, 'push2'], [4.3, 'push1'], [4.8, 'push2'], [5.6, 'push2'],
    [6.3, 'recoil', easeOut], [7.0, 'recoil'], [7.8, 'look']];
  const m = actor(c, 'man', keys, tau, 180);
  const hand = m.local(m.P.armN[1]), tq = snap2(tau);
  const contact = DUCT.x0 - 14;
  let bx = hand[0] + 92, by = hand[1] + 10;
  const k = clamp((bx + 105 - contact) / 60, 0, 1);            // how hard it is pressed into the mouth
  if (tq > 5.6) { const u = easeOut(clamp((tq - 5.6) / .6, 0, 1)); bx = lerp(bx, bx - 30, u); }
  ideaBlob(c, bx, by, {k: tq > 5.6 ? k * (1 - clamp((tq - 5.6) / .6, 0, 1)) : k, flatX: contact});
  ductMouthIn(c);
  // effort lines while straining; drawn per drawing change (on twos), fixed seeds
  if (m.P.eye < .5) { c.save(); c.strokeStyle = INK; c.lineWidth = 2.4; c.lineCap = 'round'; const hd = m.local(m.P.head); for (let q = 0; q < 3; q++) { const a = -2.4 + q * .45; c.beginPath(); c.moveTo(hd[0] + Math.cos(a) * 60, hd[1] + Math.sin(a) * 60); c.lineTo(hd[0] + Math.cos(a) * 80, hd[1] + Math.sin(a) * 80); c.stroke(); } c.restore(); }
  actor(c, 'inv', [[0, 'lens'], [4.9, 'lens'], [5.6, 'puzzled']], tau, 1770);
  if (snap2(tau) > 5.7) { const hd = [1770 - 60, 900 - 470]; write(c, '?', hd[0], hd[1], {size: 110, color: RED, weight: 700, align: 'center', p: 1}); }
  inset(c, tau, 3.4, {spread: 1, label: 'R² low', formula: writeP(tau, l1.start + 1.1, 'inadequacy = 1 − R²', 16)});
  caption(c, 'Knowledge economy: value that GAAP’s boxes miss', tau, l0.start + .2);
}
function sceneQuestion(c, tau) {
  const l = LN('question', 0);
  stageBase(c, {building: 'lab'});
  board(c, [{k: 4, c: 'b', t: -1}, {k: 1, c: 'b', t: -1}, {k: 9, c: 'b', t: -1}], tau, {pic: 'bulb', gaps: 1});
  const b1 = at('question', 0, .38), b2 = at('question', 0, .62);
  const m = actor(c, 'man', [[0, 'look'], [.5, 'look'], [1.2, 'think'], [b2 + .6, 'think'], [b2 + 1.3, 'look']], tau, 180);
  ideaBlob(c, 440, 610 + 250, {k: 0});           // the blob rests on the floor, set down during the cut
  ductMouthIn(c);
  actor(c, 'inv', [[0, 'puzzled']], tau, 1770);
  const hd = m.local(MAN.think.head);
  bubble(c, 330, 250, 330, 190, hd[0] + 20, hd[1] - 60, clamp((tau - b1 + .3) / .3, 0, 1), c => {
    write(c, 'just comply?', 330, 225, {size: 48, align: 'center', weight: 700, color: BLUE});
    c.strokeStyle = INK; c.lineWidth = 2.4; c.strokeRect(250, 250, 34, 34);
    inkLine(c, [[256, 266], [266, 280], [286, 248]], {w: 4, seed: 131, p: clamp((tau - b1 - .4) / .3, 0, 1)});
    write(c, 'the minimum', 300, 278, {size: 34, color: GREY});
  });
  bubble(c, 760, 310, 360, 190, hd[0] + 60, hd[1] - 40, clamp((tau - b2 + .3) / .3, 0, 1), c => {
    write(c, 'or say more?', 760, 285, {size: 48, align: 'center', weight: 700, color: RED});
    inkLine(c, [[700, 320], [740, 360], [760, 420]], {w: 3, color: RED, seed: 132, p: clamp((tau - b2 - .3) / .5, 0, 1)});
    write(c, '→ MD&A', 780, 350, {size: 38, color: RED, weight: 700});
  });
}
function sceneMdna(c, tau) {
  const l0 = LN('mdna', 0), l1 = LN('mdna', 1);
  stageBase(c, {building: 'lab'});
  ideaBlob(c, 250, 860, {k: 0});
  // cards travel the ribbon at constant speed from the lectern to the board
  const cards = [
    {text: 'Adjusted EBITDA*', t: l0.start + 1.0, col: RED, k: 0},
    {text: 'Organic growth*', t: l0.start + 2.2, col: RED, k: 6},
    {text: 'Free cash flow*', t: l0.start + 3.4, col: RED, k: 11},
    {text: 'We expect…', t: l1.start + .2, col: RED, k: 2, arrow: true},
    {text: 'We plan to…', t: l1.start + 1.2, col: RED, k: 3, arrow: true},
    {text: 'Next year…', t: l1.start + 2.2, col: RED, k: 5, arrow: true},
    {text: 'Our pipeline…', t: l1.start + 3.1, col: RED, k: 7, arrow: true},
    {text: 'Outlook: …', t: l1.start + 3.9, col: RED, k: 8, arrow: true},
    {text: 'R&D will…', t: l1.start + 4.6, col: RED, k: 10, arrow: true},
  ];
  const travel = 2.6;
  const tiles = [{k: 4, c: 'b', t: -1}, {k: 1, c: 'b', t: -1}, {k: 9, c: 'b', t: -1}, ...cards.map(cd => ({k: cd.k, c: 'r', t: cd.t + travel + .55, from: RIB.at(1).p}))];
  board(c, tiles, tau, {pic: 'bulb', gaps: 1});
  for (const cd of cards) {
    const u = (tau - cd.t) / travel; if (u < 0 || u > 1) continue;
    const a = RIB.at(u), s = u > .85 ? lerp(1, .5, (u - .85) / .15) : 1;
    c.save(); c.translate(...a.p); c.scale(s, s); card(c, cd.text, 0, -4, Math.atan2(a.tangent[1], a.tangent[0]) * .6, cd.col, {arrow: cd.arrow, w: cd.arrow ? 200 : 210}); c.restore();
  }
  const wk = [[0, 'write1']]; for (let t = .5; t < l1.end + .2; t += .5) wk.push([t, wk.length % 2 ? 'write2' : 'write1']);
  wk.push([l1.end + .9, 'pleased']);
  const m = actor(c, 'man', wk, tau, 420);
  pen(c, ...m.local(vadd(m.P.armN[1], [6, -8])), -1.1);
  actor(c, 'inv', [[0, 'puzzled'], [l0.start + 3.4, 'puzzled'], [l0.start + 4.2, 'lens'], [l1.end - .5, 'lens'], [l1.end + .2, 'pleased']], tau, 1770);
  write(c, 'non-GAAP measures: tailored to the firm', 90, 110, {size: 52, weight: 700, color: RED, p: writeP(tau, l0.start + 1.5, 'non-GAAP measures: tailored to the firm', 30)});
  write(c, 'forward-looking statements: where it is heading', 90, 180, {size: 52, weight: 700, color: RED, p: writeP(tau, l1.start + .3, 'forward-looking statements: where it is heading', 30)});
  write(c, '* non-GAAP', 700, 1030, {size: 28, color: GREY, p: writeP(tau, l0.start + 2, '* non-GAAP')});
}
function bars(c, x, y, title, vals, labels, tau, t0, {max, fmt, delta, deltaNote}) {
  write(c, title, x, y, {size: 50, weight: 700, p: writeP(tau, t0, title, 30)});
  const base = y + 420, bw = 130, gap = 90;
  inkLine(c, [[x - 10, base], [x + 2 * bw + gap + 60, base]], {seed: 141, w: 2.4, p: clamp((tau - t0) / .5, 0, 1)});
  vals.forEach((v, k) => {
    const g = easeOutQuint(clamp((tau - t0 - .6 - k * .6) / 1.0, 0, 1)), hgt = 320 * v / max * g, bx = x + 20 + k * (bw + gap);
    if (g <= 0) return;
    const r = rectPath(bx, base - hgt, bw, hgt);
    printFill(c, r, k ? RED_L : BLUE_L); c.save(); c.strokeStyle = k ? RED : BLUE; c.lineWidth = 3; c.stroke(r); c.restore();
    write(c, fmt(v), bx + bw / 2, base - hgt - 16, {size: 40, align: 'center', weight: 700, p: g > .95 ? 1 : 0});
    write(c, labels[k], bx + bw / 2, base + 46, {size: 32, align: 'center', color: GREY, p: g > .2 ? 1 : 0});
  });
  const q = clamp((tau - t0 - 2.0) / .5, 0, 1);
  if (q > 0) { write(c, delta, x + 2 * bw + gap + 40, y + 120, {size: 72, weight: 700, color: RED, p: q});
    if (deltaNote) write(c, deltaNote, x + 2 * bw + gap + 40, y + 170, {size: 32, color: GREY, p: q}); }
}
function sceneNumbers(c, tau) {
  const l0 = LN('numbers', 0), l1 = LN('numbers', 1);
  paper(c, PAPER, null, 5);
  write(c, 'Most adequate → least adequate statements', CX, 120, {size: 56, weight: 700, align: 'center', p: writeP(tau, .1, 'Most adequate → least adequate statements', 40)});
  bars(c, 170, 260, 'Non-GAAP talk in the MD&A', [100, 119], ['most', 'least'], tau, l0.start + .4, {max: 125, fmt: v => v === 100 ? 'index 100' : '119', delta: '+19%'});
  bars(c, 1050, 260, 'Forward-looking sentences', [30, 31.5], ['most', 'least'], tau, l1.start - .2, {max: 33, fmt: v => v === 30 ? '≈ 30' : '≈ 31.5', delta: '+5%', deltaNote: '≈ 1.5 more sentences'});
  write(c, 'Firm-years 2004–2016. Bars illustrate the reported differences; non-GAAP shown as an index.', CX, 1010, {size: 28, color: GREY, align: 'center', p: writeP(tau, l1.start + 1.5, 'x'.repeat(20), 20)});
}
function pageLines(c, x, y, w, n, seed, color = alpha(INK, .35)) {
  c.save(); c.strokeStyle = color; c.lineWidth = 5; c.lineCap = 'round';
  for (let k = 0; k < n; k++) { const L = w * (.72 + hash(k, seed) * .28) * (k % 6 === 5 ? .55 : 1); c.beginPath(); c.moveTo(x, y + k * 22); c.lineTo(x + L, y + k * 22); c.stroke(); }
  c.restore();
}
function sceneMethods(c, tau) {
  const l0 = LN('methods', 0), l1 = LN('methods', 1);
  paper(c, PAPER, null, 5);
  write(c, 'How it is measured', 90, 110, {size: 60, weight: 700, p: writeP(tau, .1, 'How it is measured', 30)});
  // page 1: keywords
  const px = 110, py = 200, pw = 470, ph = 700;
  const page = (x, t0) => { const q = clamp((tau - t0) / .5, 0, 1); if (q <= 0) return false; c.save(); c.globalAlpha *= q; printFill(c, rectPath(x, py, pw, ph), CREAM, {off: [6, 6]}); c.strokeStyle = GREY; c.lineWidth = 2; c.strokeRect(x, py, pw, ph); write(c, 'MD&A', x + 30, py + 60, {size: 44, weight: 700, color: RED}); c.restore(); return true; };
  const t0 = l0.start;
  if (page(px, t0)) {
    pageLines(c, px + 30, py + 100, pw - 60, 26, 3);
    const hits = [[3, 60, 'non-GAAP'], [9, 150, 'adjusted'], [15, 40, 'excluding'], [21, 200, 'non-GAAP']];
    hits.forEach(([row, xo, wd], k) => { const q = clamp((tau - t0 - .9 - k * .35) / .3, 0, 1); if (q <= 0) return; c.save(); c.fillStyle = alpha(OCHRE, .55); c.fillRect(px + 30 + xo - 6, py + 100 + row * 22 - 12, 150 * q, 24); c.restore(); write(c, wd, px + 30 + xo, py + 100 + row * 22 + 9, {size: 28, weight: 700, p: q > .9 ? 1 : 0}); });
    write(c, '1 · keywords → non-GAAP', px, py + ph + 60, {size: 40, weight: 700, color: RED, p: writeP(tau, t0 + .8, '1 · keywords → non-GAAP')});
  }
  // network: sentences go through convolution layers
  const nx = 700, t1 = at('methods', 0, .55);
  if (tau > t1 - .3) {
    const q = clamp((tau - t1 + .3) / .5, 0, 1); c.save(); c.globalAlpha *= q;
    for (let k = 0; k < 4; k++) { const x = nx + 40 + k * 95, hh = 300 - k * 60, pl = [[x, 520 - hh / 2], [x + 50, 520 - hh / 2 + 20], [x + 50, 520 + hh / 2 - 20], [x, 520 + hh / 2]];
      printFill(c, polyPath(pl), k === 3 ? RED_L : BLUE_L); inkLine(c, pl, {close: true, seed: 150 + k, w: 2.4, amp: .4}); }
    write(c, 'deep-learning network (CNN)', nx + 200, 330, {size: 34, color: GREY, align: 'center'});
    c.restore();
    const sents = ['We expect margins to widen next year.', 'Revenue rose 8% in 2015.', 'We plan to open 12 stores.'];
    sents.forEach((s, k) => { const u = clamp((tau - t1 - .2 - k * 1.1) / 1.0, 0, 1); if (u <= 0) return;
      const yy = 690 + k * 62, done = u >= 1, fls = k !== 1;
      write(c, s, nx - 10 + (1 - easeOut(u)) * -40, yy, {size: 32, color: INK, alpha: clamp(u * 3, 0, 1)});
      if (done) write(c, fls ? '✓ forward-looking' : '✗ not forward-looking', nx + 470, yy, {size: 32, weight: 700, color: fls ? RED : GREY});
    });
    write(c, '2 · network → forward-looking sentences', nx - 10, py + ph + 60, {size: 40, weight: 700, color: RED, p: writeP(tau, t1 + .3, '2 · network → forward-looking sentences')});
  }
  // TopicTiling: the page is cut into topic segments
  const t2 = l1.start, tx = 1330;
  if (tau > t2 - .3) {
    const q = clamp((tau - t2 + .3) / .5, 0, 1);
    const cols = {required: BLUE_L, traditional: '#d8dcc0', intangibles: OCHRE_L, other: '#e2dbd0'};
    const segs = [['required', 5], ['traditional', 6], ['intangibles', 5], ['required', 3], ['other', 3], ['intangibles', 4]];
    c.save(); c.globalAlpha *= q; printFill(c, rectPath(tx, py, pw, ph), CREAM, {off: [6, 6]}); c.strokeStyle = GREY; c.lineWidth = 2; c.strokeRect(tx, py, pw, ph); c.restore();
    let row = 0;
    segs.forEach(([g, n], k) => {
      const y0 = py + 30 + row * 24, hh = n * 24, cut = clamp((tau - t2 - .6 - k * .35) / .35, 0, 1);
      if (cut > 0) { c.save(); c.globalAlpha *= cut; c.fillStyle = cols[g]; c.fillRect(tx + 14, y0 - 4, pw - 28, hh); c.restore();
        inkLine(c, [[tx - 16, y0 - 6], [tx + pw + 16, y0 - 6]], {seed: 170 + k, w: 2, color: INK, p: cut, amp: 1.2});
        write(c, g, tx + pw - 26, y0 + hh / 2 + 10, {size: 30, weight: 700, align: 'right', color: g === 'intangibles' ? '#9a6a12' : GREY, p: cut > .8 ? 1 : 0}); }
      pageLines(c, tx + 30, y0 + 10, pw - 200, n, 20 + k, alpha(INK, .3 * q));
      row += n;
    });
    const ring = clamp((tau - at('methods', 1, .72)) / .6, 0, 1);
    if (ring > 0) {
      // circle both intangibles segments
      inkLine(c, ellPts(tx + pw - 110, py + 30 + 11 * 24 + 60, 120, 58, -.05, 28), {close: true, seed: 181, w: 4, color: RED, p: ring, amp: 2});
      inkLine(c, ellPts(tx + pw - 110, py + 30 + 23 * 24 + 48, 120, 52, .04, 28), {close: true, seed: 182, w: 4, color: RED, p: ring, amp: 2});
      write(c, 'more disclosure here', tx + 40, py + ph - 20, {size: 40, weight: 700, color: RED, p: writeP(tau, at('methods', 1, .78), 'more disclosure here')});
    }
    write(c, '3 · TopicTiling → topics', tx, py + ph + 60, {size: 40, weight: 700, color: RED, p: writeP(tau, t2 + .3, '3 · TopicTiling → topics')});
    write(c, '133 topics → 21 categories → 4 groups', tx, py + ph + 108, {size: 32, color: GREY, p: writeP(tau, t2 + 1.4, '133 topics → 21 categories → 4 groups', 36)});
  }
}
function sceneMosaic(c, tau) {
  const l = LN('mosaic', 0);
  const z = key(tau, [[0, 1], [1.2, 1], [4.2, 1.45]], easeInOutSine), fx = key(tau, [[0, CX], [1.2, CX], [4.2, 1320]], easeInOutSine), fy = key(tau, [[0, CY], [1.2, CY], [4.2, 600]], easeInOutSine);
  paper(c, PAPER, null, 5);
  cam(c, fx, fy, z);
  building(c, 'lab'); ground(c); duct(c, 1, {label: 1, stamp: 1, audit: 1}); lectern(c); ribbon(c, 1, {label: 1});
  const blues = [1, 4, 9], reds = [0, 2, 3, 5, 6, 7, 8, 10, 11];
  board(c, [...blues.map(k => ({k, c: 'b', t: -1})), ...reds.map(k => ({k, c: 'r', t: -1}))], tau, {pic: 'bulb', label: 1});
  // both channels keep carrying: blue pieces in the duct window, cards on the ribbon
  for (let k = 0; k < 3; k++) { const u = ((tau * .22 + k / 3) % 1); const a = RIB.at(u); card(c, ['We expect…', 'Adjusted EBITDA*', 'Next year…'][k], a.p[0], a.p[1] - 4, Math.atan2(a.tangent[1], a.tangent[0]) * .6, RED, {w: 200, arrow: k !== 1}); }
  actor(c, 'man', [[0, 'pleased']], tau, 420);
  actor(c, 'inv', [[0, 'pleased']], tau, 1770);
  resetT(c);
  write(c, 'a mosaic, not fragments', CX, 1010, {size: 64, weight: 700, align: 'center', p: writeP(tau, at('mosaic', 0, .35), 'a mosaic, not fragments', 26)});
}
function sceneEnd(c, tau) {
  paper(c, PAPER, null, 5);
  const lines = [
    ['When the financial statements say too little,', 70, INK, 700, .2],
    ['managers say more in the MD&A.', 70, RED, 700, 1.2],
    ['Brown, S. V., Hinson, L. A., & Tucker, J. W. (2024).', 38, GREY, 500, 2.6],
    ['Financial statement adequacy and firms’ MD&A disclosures.', 38, GREY, 500, 3.2],
    ['Contemporary Accounting Research.', 38, GREY, 500, 3.8],
  ];
  const ys = [330, 420, 640, 694, 748];
  lines.forEach(([t, s, col, wt, t0], k) => write(c, t, CX, ys[k], {size: s, color: col, weight: wt, align: 'center', p: writeP(tau, t0, t, 34)}));
  const q = clamp((tau - 2.0) / .8, 0, 1);
  inkLine(c, [[CX - 300, 520], [CX + 300, 520]], {seed: 191, w: 2, color: GREY, p: q});
  write(c, 'An illustrated summary. Figures are schematic.', CX, 960, {size: 28, color: GREY, align: 'center', p: writeP(tau, 4.4, 'An illustrated summary. Figures are schematic.', 40)});
}
const FN = {title: sceneTitle, channels: sceneChannels, fits: sceneFits, stuck: sceneStuck, question: sceneQuestion, mdna: sceneMdna, numbers: sceneNumbers, methods: sceneMethods, mosaic: sceneMosaic, end: sceneEnd};
// ---------------------------------------------------------------- score: soft cues only, the narration carries the film
function score(ac, t0, out) {
  const master = ac.createGain(); master.gain.value = .5; master.connect(out);
  let acc = 0; const start = {}; for (const s of TIMING) { start[s.id] = acc; acc += s.dur; }
  const S0 = id => start[id];
  // a slow pad under the whole film, changing chord per scene
  const roots = {title: 0, channels: 3, fits: 4, stuck: 1, question: 1, mdna: 4, numbers: 0, methods: 3, mosaic: 4, end: 0};
  for (const s of TIMING) { const r = roots[s.id]; for (const k of [0, 2, 4]) note(ac, master, pentHz(-1, r + k, 220), t0, S0(s.id) + .05, s.dur + .8, 'sine', .045); }
  // cues tied to visible actions
  const tick = (t, f = 880, g = .1) => note(ac, master, f, t0, t, .35, 'triangle', g);
  tick(S0('channels') + at('channels', 0, .5), 196, .22); noiseBurst(ac, master, t0, S0('channels') + at('channels', 0, .5), .12, .08, 3);   // stamp
  for (let k = 0; k < 12; k++) tick(S0('fits') + 3.0 + k * .12, pentHz(0, k % 5, 440), .05);
  for (const t of [2.6, 3.7, 4.8]) { noiseBurst(ac, master, t0, S0('stuck') + t, .15, .12, 7); tick(S0('stuck') + t, 110, .2); }   // bonk
  for (const t of [3.6, 4.3, 5.0]) tick(S0('stuck') + t, pentHz(0, 2, 440), .05);
  for (let t = .5; t < SC.mdna.dur - 1; t += .5) noiseBurst(ac, master, t0, S0('mdna') + t, .08, .025, 20 + t * 10);        // pen
  for (const k of [0, 1, 2, 3, 4, 5, 6, 7, 8]) tick(S0('mdna') + [1, 2.2, 3.4].concat([.2, 1.2, 2.2, 3.1, 3.9, 4.6].map(x => x + LN('mdna', 1).start - LN('mdna', 0).start))[k] + LN('mdna', 0).start + 3.15, pentHz(1, k % 5, 440), .045);
  tick(S0('end') + 1.2, pentHz(1, 0, 440), .07); tick(S0('end') + 1.45, pentHz(1, 2, 440), .06);
}
defineFilm({palette: makePalette({paper: PAPER, ink: INK}, 'pencilMinimal'), format: {ar: '16:9', width: 1920}, fps: 24, score,
  timeline: TIMING.map(s => ({name: s.id, dur: s.dur, fn: FN[s.id]}))});
</script>


--- [189] ANSWER ---
The whole film renders without errors. Now I'm checking full-size frames, starting with the push shot, which is the hardest action.

--- [229] ANSWER ---
The first full-size frames look promising. Now I'm checking the push/jam action frame by frame. The standard strip is too small to judge, so I'm writing a cropped-strip helper.

--- [235] PROMPT ---
[Image: original 3020x1820, displayed at 2000x1205. Multiply coordinates by 1.51 to map to original image.]

--- [256] PROMPT ---
[Image: original 5780x1252, displayed at 2000x433. Multiply coordinates by 2.89 to map to original image.]

--- [263] ANSWER ---
The blob now drops cleanly and the investor visibly scratches his head. Next I'm checking the remaining scenes at full size.

--- [289] EDIT:Edit ---
[Edit /home/hofmannp/animation_test/disclosure-channels/two-channels.html]
--- old ---
  printFill(c, desk, '#b88a5a');

--- new ---
  printFill(c, desk, '#b88a5a'); inkLine(c, [[540, top], [640, top - 10], [648, top + 6], [548, top + 16]], {close: true, seed: 31, w: 2.4});


--- [304] ANSWER ---
Most scenes look right now. One last layout fix: the Item 8 tab label overflows its tab on the title card. After that I'll do the full render.

--- [309] ANSWER ---
While the full render runs, I'm writing the narration mix script and checking whether a speech recogniser is installed, so I can verify pronunciation.

--- [315] ANSWER ---
No speech recogniser is installed. I'll try a small Whisper model through transformers to check how "GAAP", "MD&A" and "R squared" were spoken.

--- [324] ANSWER ---
The Whisper transcripts match the script on every line, so the pronunciations are right. Waiting for the render now.

--- [337] PROMPT ---
[Image: original 1440x5168, displayed at 557x2000. Multiply coordinates by 2.59 to map to original image.]

--- [344] ANSWER ---
I've made the film: a narrated, hand-drawn explainer of the paper's two-channel idea, 112 seconds long.

**Watch:** `disclosure-channels/two-channels-narrated.mp4`. It's 1920×1080 at 24 fps, has a soft music bed that drops under the voice, and carries English subtitles as a separate track you can turn off. The version with music only is `out/two-channels-final.mp4`.

**How the story runs:**
1. **Title:** a 10-K with two tabs, Item 7 (MD&A) and Item 8 (financial statements).
2. **Two channels:** a manager on the left, an investor on the right. Between them run a rigid blue duct (the financial statements, stamped "GAAP rules" and "audited") and a flexible red paper ribbon (the MD&A). The investor holds an empty 4×3 board labelled "the investor's picture".
3. **Old economy:** a factory firm pushes a crate straight through the duct. The board fills with blue tiles, and a small chart shows price tightly explained by earnings and book value ("R² high").
4. **Knowledge economy:** the manager tries to push a soft "ideas, software, brands" blob into the square duct. It jams; he strains twice, only three tiles get through, and the investor scratches his head. The chart scatters ("R² low") and "inadequacy = 1 − R²" appears.
5. **The question:** thought bubbles ask "just comply?" or "say more? → MD&A".
6. **The turn:** he writes at a lectern. Non-GAAP cards and forward-looking cards ("We expect…") travel the ribbon and fill the gaps in red.
7. **Findings:** non-GAAP discussion up 19%; forward-looking sentences up about 5%, roughly 1.5 more out of about 30.
8. **How it was measured:** keyword highlighting, the neural network sorting sentences into forward-looking or not, and TopicTiling cutting an MD&A into topics, with the intangibles segments circled.
9. **Mosaic:** the board ends as a mix of blue and red tiles, captioned "a mosaic, not fragments", followed by a citation card.

**Checks:**
- **File:** the final file has all 2,687 frames and decodes without errors.
- **Voice:** it uses your cloned voice in x-vector mode. I ran every line through Whisper, and the transcripts match the script, including "GAAP", "MD&A" and "R squared".
- **Stillness:** about two-thirds of frames are pixel-identical to the one before, so nothing jitters, shakes or flickers while it stands still.
- **What I looked at:** stills of every scene and frame-by-frame strips of the push, the jam and the blob dropping. I have not watched it at normal speed with sound, so the pacing and the audio mix still need your viewing.

**Things you might want changed:**
- The bars in the findings scene are schematic. Non-GAAP is shown as an index (100 vs 119), and the forward-looking bars are about 30 vs 31.5, built from the paper's +1.508 (5.1%). An on-screen footnote says so.
- In the writing scene, the manager's head slightly overlaps the "Item 8" label.
- Between the question and writing scenes the blob moves from the floor by the duct to beside the building. It happens on a cut, but it's a small continuity jump.

**To edit:** change the narration in `narration.json`, then run `python voice.py`, `node render.mjs two-channels.html` and `sh mix.sh`. Scene lengths follow the spoken lines automatically. The handwriting font, Caveat, is embedded under its open font licence. The rendered frames in `out/two-channels-frames/` take up a lot of disk and can be deleted.

Nothing is committed yet.

--- [346] PROMPT ---
that two-channels.mp4 does not have voice?

--- [357] ANSWER ---
That's expected: `two-channels.mp4` is the silent picture-only file. The voice is in a different file.

| File | What's in it |
|---|---|
| `out/two-channels.mp4` | picture only, no sound |
| `out/two-channels-final.mp4` | picture plus the soft music, no voice |
| **`disclosure-channels/two-channels-narrated.mp4`** | picture, music, voice and subtitles |

The narrated file sits in the `disclosure-channels/` folder itself, not in `out/`. I measured the audio in both sound files: the music-only one is quiet (about −39 dB average), while the narrated one is much louder (about −18 dB), which is the voice.

--- [359] PROMPT ---
use python to gen some background music to fit in, and render again

--- [375] EDIT:Write ---
[Write /home/hofmannp/animation_test/disclosure-channels/music.py]
"""Background score for "Two Channels", synthesized with numpy.

A warm, unhurried piano-and-pad piece in D (84 BPM) whose harmony follows the
story: steady for the two channels, bright when the crate fits, minor and sparse
when the idea jams, suspended on the question, resolving when the MD&A carries
the rest, full for the mosaic. Soft cues sit on the visible actions
(stamp, bonks, tiles landing, pen). Scene timing comes from timing.js.

    python music.py          # writes out/music.wav
"""
import json
import re
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, sosfilt

HERE = Path(__file__).resolve().parent
SR = 48000
BPM = 84
BEAT = 60 / BPM
rng = np.random.default_rng(7)

TIMING = json.loads(re.search(r"const TIMING = (\[.*\]);", (HERE / "timing.js").read_text(), re.S).group(1))
START, t = {}, 0.0
for s in TIMING:
    START[s["id"]] = t
    t += s["dur"]
TOTAL = t
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


# ---------------------------------------------------------------- instruments
def piano(m, dur=2.8, vel=1.0):
    f, t = hz(m), tv(dur)
    x = sum(a * np.sin(2 * np.pi * f * k * (1 + 0.0004 * k * k) * t) * np.exp(-t * (1.3 + 0.9 * k))
            for k, a in enumerate([1, .45, .22, .12, .06], start=1))
    x *= np.minimum(1, t / 0.004)
    return x * vel * 0.3


def pad(ms, dur, att=1.4, rel=1.6):
    t = tv(dur + rel)
    x = np.zeros_like(t)
    for m in ms:
        for det in (-0.07, 0.0, 0.08):
            f = hz(m) * 2 ** (det / 12)
            x += np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t + 1.1)
    env = np.minimum(1, t / att) * np.where(t > dur, np.exp(-(t - dur) / (rel / 3)), 1)
    return lowpass(x * env, 1800) * 0.035


def bass(m, dur=1.6):
    f, t = hz(m), tv(dur)
    x = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t)
    return x * np.exp(-t * 2.2) * np.minimum(1, t / 0.01) * 0.28


def chime(m, dur=1.6, vel=1.0):
    f, t = hz(m), tv(dur)
    x = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t * 6)
    return x * np.exp(-t * 3.2) * np.minimum(1, t / 0.002) * 0.16 * vel


def brush(dur=0.12, vel=1.0):
    t = tv(dur)
    return highpass(rng.standard_normal(len(t)), 3500) * np.exp(-t * 40) * 0.03 * vel


def thud(vel=1.0):
    t = tv(0.5)
    f = 70 * np.exp(-t * 6) + 45
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)
    x += lowpass(rng.standard_normal(len(t)), 600) * np.exp(-t * 30) * 0.4
    return x * 0.45 * vel


def stamp():
    t = tv(0.35)
    return (lowpass(rng.standard_normal(len(t)), 1200) * np.exp(-t * 25) * 0.5
            + np.sin(2 * np.pi * 150 * t) * np.exp(-t * 18) * 0.35)


def scratch(dur=0.35):
    t = tv(dur)
    n = sosfilt(butter(2, [2500, 7000], "bandpass", fs=SR, output="sos"), rng.standard_normal(len(t)))
    return n * (0.5 + 0.5 * np.sin(2 * np.pi * 11 * t) ** 2) * np.sin(np.pi * t / dur) * 0.02


def reverb(x, secs=2.4, wet=0.22):
    t = tv(secs)
    ir = rng.standard_normal(len(t)) * np.exp(-t / 0.5)
    ir = lowpass(ir, 5000)
    ir /= np.sqrt(np.sum(ir ** 2))
    y = fftconvolve(x, ir)[: len(x)]
    return x * (1 - wet) + y * wet * 0.9


# ---------------------------------------------------------------- harmony per scene (MIDI roots of 4-beat bars)
D, Bm, G, A, Em, Fsm, Asus = [50, 57, 62, 66], [47, 54, 59, 62], [43, 55, 59, 62], [45, 57, 61, 64], [40, 55, 59, 64], [42, 57, 61, 66], [45, 57, 62, 64]
PLAN = {   # scene: (chord cycle, texture)
    "title":    ([D, G, D, A], "piano"),
    "channels": ([D, Bm, G, A], "pulse"),
    "fits":     ([D, G, A, D], "bright"),
    "stuck":    ([Bm, Em, Bm, Fsm], "sparse"),
    "question": ([Asus, Asus, Em, Asus], "held"),
    "mdna":     ([G, A, D, Bm], "flow"),
    "numbers":  ([D, Bm, G, A], "pulse"),
    "methods":  ([Bm, G, D, A], "pulse"),
    "mosaic":   ([G, A, Fsm, Bm], "flow"),
    "end":      ([G, A, D, D], "piano"),
}
MELODY = {  # scale-degree motifs (MIDI offsets over the chord's third voice), one per bar
    "piano":  [[0, None, 2, None], [4, None, 2, None]],
    "bright": [[0, 2, 4, 7], [4, 2, 0, None]],
    "flow":   [[0, 4, 7, 4], [2, 5, 9, 5]],
    "held":   [[None, None, None, None]],
    "sparse": [[0, None, None, None], [None, None, -1, None]],
    "pulse":  [[0, None, None, 2], [None, None, 4, None]],
}


def build():
    n = int((TOTAL + 4) * SR)
    mus, fx = np.zeros(n), np.zeros(n)

    def put(buf, t0, x, gain=1.0):
        a = int(t0 * SR)
        if a >= n:
            return
        x = x[: n - a]
        buf[a:a + len(x)] += x * gain

    bar = 4 * BEAT
    for s in TIMING:
        sid, s0, dur = s["id"], START[s["id"]], s["dur"]
        chords, tex = PLAN[sid]
        nbars = max(1, int(np.ceil(dur / bar)))
        for b in range(nbars):
            tb = s0 + b * bar
            if tb >= s0 + dur - 0.2:
                break
            ch = chords[b % len(chords)]
            remain = min(bar, s0 + dur - tb)
            put(mus, tb, pad(ch[1:], remain, att=0.9 if b else 1.6), 1.0 if tex != "sparse" else 0.8)
            if tex in ("pulse", "flow", "bright", "piano"):
                put(mus, tb, bass(ch[0]), 1.0)
                if tex != "piano":
                    put(mus, tb + 2 * BEAT, bass(ch[0] + (7 if tex == "bright" else 0)), .7)
            if tex == "sparse":
                put(mus, tb, bass(ch[0] - 12 + 12), .8)
            if tex in ("pulse", "flow"):                       # soft brushes on the off-beats
                for k in range(4):
                    put(mus, tb + (k + .5) * BEAT, brush(vel=1.0 if k % 2 else .6))
            if tex == "flow":                                  # eighth-note arpeggio
                arp = ch[1:] + [ch[2] + 12]
                for k in range(8):
                    if tb + k * BEAT / 2 < s0 + dur - .2:
                        put(mus, tb + k * BEAT / 2, piano(arp[k % 4] + 12, 1.2, .35))
            motif = MELODY[tex][b % len(MELODY[tex])]
            for k, deg in enumerate(motif):
                tn = tb + k * BEAT
                if deg is None or tn > s0 + dur - .4:
                    continue
                put(mus, tn, piano(ch[2] + 12 + deg, 2.6, .8 if tex != "sparse" else .6))
            if tex == "held":                                  # a quiet clock while he thinks
                for k in range(4):
                    put(fx, tb + k * BEAT, chime(96 if k % 2 else 91, .25, .25))

    # the final chord rings out
    put(mus, START["end"] + SC["end"]["dur"] - 3.0, sum(piano(m + 12, 4, .6) for m in D[1:]))

    # ---- cues on the visible actions (times mirror two-channels.html)
    put(fx, START["channels"] + line_at("channels", 0, .5) - START["channels"], stamp(), .9)
    for k in range(12):
        put(fx, START["fits"] + 3.0 + k * .12, chime(74 + [0, 2, 4, 7, 9][k % 5] + 12 * (k // 5 % 2), vel=.7))
    for tb in (2.6, 3.7, 4.8):
        put(fx, START["stuck"] + tb, thud(), 1.0)
    for tb in (3.6, 4.3, 5.0):
        put(fx, START["stuck"] + tb, chime(81, vel=.6))
    put(fx, START["stuck"] + 6.1, thud(.6), .7)                  # the blob lands on the floor
    l0, l1 = SC["mdna"]["lines"]
    card_t = [l0["start"] + x for x in (1.0, 2.2, 3.4)] + [l1["start"] + x for x in (.2, 1.2, 2.2, 3.1, 3.9, 4.6)]
    for tb in np.arange(0.0, l1["end"] + .2, .5):
        put(fx, START["mdna"] + tb, scratch(), 1.0)
    for k, tc in enumerate(card_t):
        put(fx, START["mdna"] + tc + 2.6 + .55, chime(78 + [0, 2, 4, 7, 9][k % 5], vel=.7))

    out = reverb(mus, wet=.25) + reverb(fx, wet=.15)
    fade_in, fade_out = int(1.0 * SR), int(3.0 * SR)
    out[:fade_in] *= np.linspace(0, 1, fade_in)
    end = int(TOTAL * SR)
    out[end - fade_out:end] *= np.linspace(1, 0, fade_out)
    out = out[:end]
    out = out / np.max(np.abs(out)) * 0.8
    stereo = np.stack([out, np.roll(out, int(.011 * SR)) * .97 + out * .03], axis=1)
    (HERE / "out").mkdir(exist_ok=True)
    wavfile.write(HERE / "out" / "music.wav", SR, (stereo * 32767).astype(np.int16))
    print(f"out/music.wav  {TOTAL:.2f} s")


if __name__ == "__main__":
    build()


--- [384] ANSWER ---
Next I'm removing the in-browser score from the film, pointing the mix at the new music, then generating the music and re-rendering.

--- [387] PROMPT ---
[Request interrupted by user for tool use]

--- [388] PROMPT ---
I want the background music to be more light and joyful

--- [393] EDIT:Write ---
[Write /home/hofmannp/animation_test/disclosure-channels/music.py]
"""Background score for "Two Channels", synthesized with numpy.

Light and joyful: D major, 104 BPM. Bouncy marimba off-beat chords, a plucked
ukulele-like strum, glockenspiel tunes, shaker and soft claps. No minor chords;
the jam is played as comic tip-toe pizzicato, the question as a playful tick-tock,
the MD&A and mosaic as the happiest, fullest sections. Soft cues sit on the
visible actions (stamp, bonks, tiles landing, pen). Timing comes from timing.js.

    python music.py          # writes out/music.wav
"""
import json
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
    """Karplus-Strong string: a bright ukulele-ish pluck."""
    f = hz(m)
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
    return (x + kick(.8)[: len(t)].tolist() + [0] * 0 if False else x + np.pad(kick(.8), (0, len(t) - len(kick())))) * vel


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


--- [402] ANSWER ---
Two quick fixes before running: a messy line in `boing()`, and caching for the slow string-pluck synth.

--- [409] ANSWER ---
The music generated in 4 seconds. Now I'm re-rendering the film and mixing it with the new music and the narration.

--- [423] ANSWER ---
The narrated film now has new, lighter background music: `disclosure-channels/two-channels-narrated.mp4`.

**What the music does:**
- **Overall:** it's in D major at 104 beats per minute, with no minor chords. Glockenspiel tunes play over off-beat marimba chords, a plucked ukulele-like strum, a shaker and soft claps.
- **By scene:** the intro and ending are gentle plucked arpeggios. The channels, findings and methods scenes get a light, bouncy groove. The old-economy scene, the MD&A writing scene and the mosaic are the fullest and happiest, with strums and claps. The jam and the question are playful tip-toe pizzicato rather than anything sad.
- **Sound effects:** each follows something on screen: the GAAP stamp, cartoon "boing" bonks on each push, glockenspiel notes as tiles land on the board, and pen scratches while the manager writes.

**How it's mixed:** the music sits about 6–7 dB below the voice and dips further whenever the narrator speaks. The file is 111.96 s with picture, audio and the subtitle track, and it decodes without errors. I checked the levels by measurement only and haven't listened to it, so the music-to-voice balance still needs your ear.

**Other changes:** I removed the old browser-generated cue sounds from the film so they don't clash with the new score. The file with music but no voice (`out/two-channels-final.mp4`) no longer exists, and the old `mix.sh` would break without it, so I rewrote it too.

**To tweak:** edit `music.py` and run `python music.py && sh mix.sh`; the video doesn't need re-rendering. To change the music level, edit `volume=0.6` in `mix.sh`.

--- [426] PROMPT ---
I will be posthing this video to linkedin 
help me polish my post. I want to be short, brief, and professional

```
claude opus 5.5 is crazy. 
It creates a video to explain a CAR paper with hand-drawing animations with html and js code, then converted it to a mp4 file. Zero shot. 
I am so amazed by the progress of AI.
```

--- [430] ANSWER ---
Here's a polished version:

```
I asked Claude Opus 5.5 to explain a Contemporary Accounting Research paper as a short animated video, and it built one in a single working session.

It wrote the hand-drawn animation in HTML/JavaScript, created the narration and background music, and rendered everything to MP4.

The paper: Brown, Hinson & Tucker (2024), "Financial statement adequacy and firms' MD&A disclosures." When financial statements capture a firm's value poorly, managers say more in the MD&A, through non-GAAP measures and forward-looking statements.

Remarkable progress in what AI can do for research communication.

#AI #Accounting #ResearchCommunication #GenerativeAI
```

What I changed:
- **"Zero shot":** I dropped it because it isn't quite accurate. I checked frames and fixed problems along the way, and you asked for the music to be redone. "A single working session" is true, and still impressive.
- **Credit and substance:** naming the paper and its authors, plus one line on the finding, gives professional readers something to take away beyond the AI angle. Consider tagging the authors, since they may want to share it.
- **Tone:** "crazy" and "so amazed" are replaced with a calmer line that fits LinkedIn.

A shorter version, if you prefer:

```
Claude Opus 5.5 turned a Contemporary Accounting Research paper into a narrated, hand-drawn explainer video, writing the animation code, voice-over and music itself.

Paper: Brown, Hinson & Tucker (2024), "Financial statement adequacy and firms' MD&A disclosures."
```

--- [432] PROMPT ---
we also want to say the TTS is by qwen run locally

--- [435] ANSWER ---
```
I asked Claude Opus 5.5 to explain a Contemporary Accounting Research paper as a short animated video, and it built one in a single working session.

It wrote the hand-drawn animation in HTML/JavaScript, composed the background music, and rendered everything to MP4. The narration was generated locally with Qwen3-TTS, an open-source text-to-speech model.

The paper: Brown, Hinson & Tucker (2024), "Financial statement adequacy and firms' MD&A disclosures." When financial statements capture a firm's value poorly, managers say more in the MD&A, through non-GAAP measures and forward-looking statements.

Remarkable progress in what AI can do for research communication.

#AI #Accounting #ResearchCommunication #GenerativeAI #OpenSource
```

Shorter version:

```
Claude Opus 5.5 turned a Contemporary Accounting Research paper into a narrated, hand-drawn explainer video, writing the animation code and music itself. The voice-over was generated locally with Qwen3-TTS.

Paper: Brown, Hinson & Tucker (2024), "Financial statement adequacy and firms' MD&A disclosures."
```

The narration uses a cloned voice (`ref_clone.wav`). If that isn't your own voice, check you have the person's permission and add a line saying the voice is AI-generated.

--- [439] PROMPT ---
/model

--- [456] ANSWER ---
The forked `/wrap-session` finished and committed the film as `5120e05`, 22 files. Nothing was pushed because the repo has no remote. I haven't checked the commit myself.

- **Committed:** everything under `disclosure-channels/`, including `two-channels-narrated.mp4`, the scripts to regenerate it, and this session's journal and prompt records.
- **Left out on purpose:** rendered frames, per-line voice clips, intermediate files and `node_modules`, all covered by a new `disclosure-channels/.gitignore`.
- **Not from this session:** `blender/`, `coanimator/`, `film/out/zh/`, `film/tts_line.py` and the `.gitignore` change are still untracked or modified.

**Still open before you post:**
1. Listen to the narrated MP4 with sound. I only measured the music-to-voice balance and haven't heard it. The level is `volume=0.6` in `mix.sh`.
2. In the writing scene the manager's head slightly overlaps the "Item 8" label.
3. The idea blob jumps position on the cut between the question and writing scenes.
4. Confirm you have permission to use the cloned voice reference (`film/voice/ref_clone.wav`), and consider saying in the post that the voice is AI-generated.

I can fix items 2 and 3 and re-render if you want.

--- [459] PROMPT ---
/compact

