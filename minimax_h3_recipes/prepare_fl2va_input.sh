#!/usr/bin/env bash
set -Eeuo pipefail

H3_SOURCE_VIDEO="/workspace/sglang/outputs/minimax_h3_recipes/t2va_white_cat_5s.mp4"
H3_INPUT_DIR="/workspace/sglang/inputs/minimax_h3_recipes"
H3_FIRST_FRAME="${H3_INPUT_DIR}/t2va_first_frame.png"
H3_PYTHON_BIN="${H3_PYTHON_BIN:-/workspace/.sglang/bin/python}"

if [[ ! -s "${H3_SOURCE_VIDEO}" ]]; then
  echo "Missing T2VA source video: ${H3_SOURCE_VIDEO}" >&2
  echo "Run ./run_t2va.sh first." >&2
  exit 2
fi

H3_FFMPEG_BIN="${H3_FFMPEG_BIN:-$("${H3_PYTHON_BIN}" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')}"
if [[ ! -x "${H3_FFMPEG_BIN}" ]]; then
  echo "ffmpeg executable not found: ${H3_FFMPEG_BIN}" >&2
  exit 2
fi

mkdir -p "${H3_INPUT_DIR}"
"${H3_FFMPEG_BIN}" \
  -hide_banner \
  -loglevel error \
  -y \
  -i "${H3_SOURCE_VIDEO}" \
  -frames:v 1 \
  "${H3_FIRST_FRAME}"

test -s "${H3_FIRST_FRAME}"
echo "FL2VA first-frame input: ${H3_FIRST_FRAME}"
