#!/bin/sh
# Picture + numpy score (music.py) + narration (voice.py). The music ducks under
# the voice; subtitles ride along as a text track.
#   node render.mjs two-channels.html && python music.py && sh mix.sh
set -e
cd "$(dirname "$0")"
V=out/voice
ffmpeg -y -loglevel error -i out/two-channels.mp4 -i out/music.wav -i $V/narration.wav -i $V/narration.srt \
  -filter_complex "[2:a]aresample=48000,aformat=channel_layouts=stereo,highpass=f=70,apad,asplit=2[vo][key];\
[1:a]aformat=channel_layouts=stereo,volume=0.6[m];\
[m][key]sidechaincompress=threshold=0.02:ratio=3:attack=20:release=500[bed];\
[bed][vo]amix=inputs=2:weights=1 1.8:normalize=0:duration=first,alimiter=limit=0.85:level=0[mix]" \
  -map 0:v -map "[mix]" -map 3:s -c:v copy -c:a aac -b:a 192k -c:s mov_text \
  -metadata:s:s:0 language=eng -movflags +faststart two-channels-narrated.mp4
echo "wrote two-channels-narrated.mp4"
