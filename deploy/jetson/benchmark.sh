#!/usr/bin/env bash
# benchmark.sh — ukur latency, throughput (FPS), GPU memory, power di Jetson AGX Orin.
# Membandingkan FP32 vs FP16. WAJIB di Jetson.
#
# Metrik skripsi (bagian Deployment): latency/frame (ms), FPS, GPU mem (MB),
# power (W via tegrastats), FP32 vs FP16.
#
# Usage:  bash benchmark.sh <engine_fp32> <engine_fp16>  [iterations]
set -e
E32="${1:?Usage: benchmark.sh <engine_fp32> <engine_fp16> [iters]}"
E16="${2:?butuh engine fp16}"
ITERS="${3:-300}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
SHAPES="voxels:10000x32x4,voxel_num:10000,voxel_idxs:10000x4"
LOG=bench_$(date +%Y%m%d_%H%M%S); mkdir -p "$LOG"

# Set daya MAX + jetson_clocks agar angka stabil & reproducible (catat mode di laporan)
echo "Set power mode MAXN + jetson_clocks (butuh sudo)..."
sudo nvpmodel -m 0 || true
sudo jetson_clocks || true

run_one() {  # $1=engine $2=tag
  echo "=== $2 ==="
  # tegrastats jalan di background utk rekam power+mem
  ( sudo tegrastats --interval 100 > "$LOG/tegrastats_$2.txt" ) &
  TGPID=$!
  sleep 1
  # trtexec: latency & throughput engine
  "$TRTEXEC" --loadEngine="$1" --shapes="$SHAPES" \
    --iterations="$ITERS" --avgRuns="$ITERS" --noDataTransfers --useSpinWait \
    > "$LOG/trtexec_$2.txt" 2>&1 || true
  sleep 1; sudo kill $TGPID 2>/dev/null || true
  # ringkas
  echo "  --- latency/throughput ($2) ---"
  grep -aE "Latency:|Throughput:|GPU Compute Time" "$LOG/trtexec_$2.txt" | sed 's/^/    /'
  # power: kolom VDD_GPU_SOC / POM_5V_GPU tergantung board; ambil rata2 angka mW di baris
  echo "  --- power/mem (tegrastats $2, rata-rata) ---"
  awk '/RAM/ {for(i=1;i<=NF;i++){if($i=="RAM"){print $(i+1)}}}' "$LOG/tegrastats_$2.txt" | head -1 | sed 's/^/    RAM sample: /'
  grep -aoE "VDD_(GPU_SOC|CPU_CV|IN)[^ ]* [0-9]+mW" "$LOG/tegrastats_$2.txt" 2>/dev/null | \
    awk '{split($2,a,"m"); s[$1]+=a[1]; n[$1]++} END{for(k in s) printf "    %s avg %.0f mW\n",k,s[k]/n[k]}' || echo "    (parse power manual dari $LOG/tegrastats_$2.txt)"
}

run_one "$E32" FP32
run_one "$E16" FP16

echo; echo "=== RINGKAS (isi tabel skripsi dari $LOG/) ==="
printf "%-8s %-14s %-14s %-12s\n" "presisi" "latency(ms)" "throughput(fps)" "catatan"
for tag in FP32 FP16; do
  lat=$(grep -aE "GPU Compute Time" "$LOG/trtexec_$tag.txt" | grep -aoE "mean = [0-9.]+" | head -1 | awk '{print $3}')
  thr=$(grep -aE "Throughput:" "$LOG/trtexec_$tag.txt" | grep -aoE "[0-9.]+ qps" | head -1 | awk '{print $1}')
  printf "%-8s %-14s %-14s %-12s\n" "$tag" "${lat:-?}" "${thr:-?}" "lihat tegrastats_$tag.txt utk power/mem"
done
echo "Log lengkap: $LOG/"
