#!/usr/bin/env bash
# build_engine.sh — bangun TensorRT engine (FP32 & FP16) dari ONNX PointPillars di Jetson AGX Orin.
# WAJIB dijalankan DI JETSON (engine spesifik-arsitektur; tak bisa dibangun di x86).
# Prasyarat: JetPack >=5.0 (TensorRT 8.x). trtexec ada di /usr/src/tensorrt/bin/trtexec.
#
# Usage:  bash build_engine.sh /path/ke/pointpillar_m300.onnx  [out_dir]
set -e
ONNX="${1:?Usage: build_engine.sh <onnx> [out_dir]}"
OUT="${2:-$(dirname "$ONNX")}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
mkdir -p "$OUT"
BASE="$OUT/$(basename "${ONNX%.onnx}")"

echo "=== TensorRT engine build (Jetson) ==="
"$TRTEXEC" --version | head -1 || true

# Input shape fixed (voxels 10000x32x4, num 10000, idxs 10000x4). Kalau ONNX punya dim dinamis,
# trtexec pakai --shapes; sesuaikan bila perlu.
SHAPES="voxels:10000x32x4,voxel_num:10000,voxel_idxs:10000x4"

echo; echo "--- [1/2] FP32 ---"
"$TRTEXEC" --onnx="$ONNX" --saveEngine="${BASE}_fp32.engine" \
  --shapes="$SHAPES" --workspace=4096 --verbose 2>&1 | tail -20

echo; echo "--- [2/2] FP16 ---"
"$TRTEXEC" --onnx="$ONNX" --saveEngine="${BASE}_fp16.engine" \
  --shapes="$SHAPES" --fp16 --workspace=4096 --verbose 2>&1 | tail -20

echo; echo "=== SELESAI ==="
ls -la "${BASE}_fp32.engine" "${BASE}_fp16.engine"
echo "Lanjut: bash benchmark.sh untuk ukur latency/FPS/power."
