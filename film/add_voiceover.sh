#!/bin/sh
# Mix the narration over the film: the music/foley bed ducks under the voice
# (sidechain compression), subtitles ride along as a toggleable text track.
#   sh film/add_voiceover.sh
set -e
cd "$(dirname "$0")/.."
V=film/out/voice

ffmpeg -y -loglevel error -stats \
  -i through-claudes-eyes.mp4 \
  -i "$V/narration.wav" \
  -i "$V/narration.srt" \
  -filter_complex "\
[1:a]aresample=44100,aformat=channel_layouts=stereo,highpass=f=70,asplit=2[vo][key];\
[0:a][key]sidechaincompress=threshold=0.015:ratio=5:attack=15:release=500[bed];\
[bed][vo]amix=inputs=2:weights=0.6 1.9:normalize=0,alimiter=limit=0.8:level=0[mix]" \
  -map 0:v -map "[mix]" -map 2:s \
  -c:v copy -c:a aac -b:a 192k -c:s mov_text \
  -metadata:s:a:0 title="Narrated mix" -metadata:s:s:0 language=eng \
  -movflags +faststart \
  through-claudes-eyes-narrated.mp4

echo "wrote through-claudes-eyes-narrated.mp4"
