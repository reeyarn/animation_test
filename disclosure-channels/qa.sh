#!/bin/sh
# Cropped consecutive-frame strip: sh qa.sh START COUNT STEP X Y W H OUT.jpg
set -e
S=$1; N=$2; ST=$3; X=$4; Y=$5; W=$6; H=$7; OUT=$8
D=$(mktemp -d)
L=$(seq -s, $S $ST $((S + (N-1)*ST)))
node render.mjs two-channels.html --only $L --out $D >/dev/null
ffmpeg -v error -y -pattern_type glob -i "$D/two-channels-frames/*.png" -vf "crop=$W:$H:$X:$Y,scale=iw/2:-2,drawtext=text='%{frame_num}':start_number=0:x=6:y=6:fontsize=18:fontcolor=black,tile=6x$(( (N+5)/6 )):padding=4" -frames:v 1 "$OUT"
rm -rf $D
