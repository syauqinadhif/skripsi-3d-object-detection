"""
select_ugm_frames.py — Seleksi 500 frame dari dataset UGM untuk labeling

Strategi:
  1. Baca semua meta YAML (GPS, velocity, timestamp)
  2. Filter frame yang berhenti (v < 0.5 m/s) — tidak informatif untuk deteksi objek
  3. Pilih frame menggunakan distance-based sampling:
     frame berikutnya dipilih hanya jika kendaraan sudah bergerak >= min_dist meter
     dari frame terakhir yang dipilih → distribusi spasial & temporal otomatis merata
  4. Jika hasil < target, turunkan threshold; jika > target, naikkan threshold
  5. Output:
     - selected_frames.txt  : daftar frame_id yang dipilih
     - selection_map.png    : peta GPS rute + frame yang dipilih (visual)

Usage:
    python scripts/select_ugm_frames.py
    python scripts/select_ugm_frames.py --target 500 --min_vel 0.5 --output_dir output/ugm_selection
    python scripts/select_ugm_frames.py --source_dir "/path/to/route" --target 300
"""

import argparse
import os
import math
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_SOURCE = "/media/mf/AUTODRIVING-4TB/UGM Baru/autoriving-oskarnatan/datasetx/2026-02-09_route00"
DEFAULT_OUTPUT = "/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi/output/ugm_selection"
DEFAULT_TARGET = 500
DEFAULT_MIN_VEL = 0.5   # m/s — frame di bawah ini dianggap berhenti


# ── Geo helpers ───────────────────────────────────────────────────────────────

def haversine_m(lat1, lon1, lat2, lon2):
    """Jarak (meter) antara dua koordinat GPS."""
    R = 6_371_000  # radius bumi (m)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ── Core ──────────────────────────────────────────────────────────────────────

def load_meta(source_dir: Path):
    """Baca semua meta YAML, return list of dict sorted by timestamp."""
    meta_dir = source_dir / "meta"
    files = sorted(meta_dir.glob("*.yml"))
    records = []
    print(f"  Reading {len(files)} meta files...")
    for f in tqdm(files, desc="  Loading meta"):
        with open(f) as fp:
            m = yaml.safe_load(fp)
        frame_id = f.stem          # e.g. '1762315221_0199838640'
        lat, lon = m['global_position_latlon']
        vel = float(m['velocity'])
        sec = int(m['sec'])
        records.append({
            'frame_id': frame_id,
            'lat': lat,
            'lon': lon,
            'vel': vel,
            'sec': sec,
        })
    return records


def distance_based_select(records, min_vel, target, verbose=True):
    """
    Distance-based frame selection.
    Cari min_dist (meter) sehingga jumlah frame terpilih ~= target.
    """
    # Filter berhenti
    moving = [r for r in records if r['vel'] >= min_vel]
    if verbose:
        print(f"\n  Total frames  : {len(records)}")
        print(f"  Moving frames : {len(moving)} (v >= {min_vel} m/s)")

    # Estimasi total jarak tempuh
    total_dist = 0
    for i in range(1, len(moving)):
        total_dist += haversine_m(
            moving[i-1]['lat'], moving[i-1]['lon'],
            moving[i]['lat'],   moving[i]['lon']
        )
    if verbose:
        print(f"  Rute total    : {total_dist/1000:.2f} km")
        print(f"  Estimasi gap  : {total_dist/target:.1f} m/frame untuk {target} frame")

    # Binary search untuk min_dist yang menghasilkan ~target frame
    lo, hi = 1.0, total_dist
    best_selected = []
    for _ in range(50):
        mid = (lo + hi) / 2
        selected = _select_with_gap(moving, mid)
        if len(selected) > target:
            lo = mid
        else:
            hi = mid
        best_selected = selected
        if abs(len(selected) - target) <= 5:
            break

    # Fine tune: jika masih > target, trim secara even
    if len(best_selected) > target:
        idx = np.round(np.linspace(0, len(best_selected) - 1, target)).astype(int)
        best_selected = [best_selected[i] for i in idx]

    return best_selected, total_dist


def _select_with_gap(records, min_dist_m):
    """Pilih frame jika jarak dari frame terakhir yang dipilih >= min_dist_m."""
    selected = [records[0]]
    for r in records[1:]:
        last = selected[-1]
        dist = haversine_m(last['lat'], last['lon'], r['lat'], r['lon'])
        if dist >= min_dist_m:
            selected.append(r)
    return selected


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_selection(all_records, selected, output_path: Path, min_vel):
    """Buat peta GPS: semua frame (abu) + selected (hijau) + stopped (merah)."""
    moving  = [r for r in all_records if r['vel'] >= min_vel]
    stopped = [r for r in all_records if r['vel'] < min_vel]

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    # Semua frame moving (rute)
    lons_m = [r['lon'] for r in moving]
    lats_m = [r['lat'] for r in moving]
    ax.plot(lons_m, lats_m, '-', color='#444466', linewidth=0.8, alpha=0.5, label='Rute (moving)')

    # Frame berhenti
    if stopped:
        lons_s = [r['lon'] for r in stopped]
        lats_s = [r['lat'] for r in stopped]
        ax.scatter(lons_s, lats_s, c='#ff4444', s=4, alpha=0.4, label=f'Berhenti ({len(stopped)})')

    # Frame terpilih
    lons_sel = [r['lon'] for r in selected]
    lats_sel = [r['lat'] for r in selected]
    ax.scatter(lons_sel, lats_sel, c='#00ff88', s=18, alpha=0.9, zorder=5,
               label=f'Dipilih ({len(selected)})')

    # Start / end marker
    ax.plot(moving[0]['lon'],  moving[0]['lat'],  'w^', markersize=10, zorder=6, label='Start')
    ax.plot(moving[-1]['lon'], moving[-1]['lat'], 'ws', markersize=10, zorder=6, label='End')

    ax.set_xlabel('Longitude', color='white')
    ax.set_ylabel('Latitude',  color='white')
    ax.tick_params(colors='white')
    ax.set_title(f'UGM Dataset — Frame Selection\n{len(selected)} frames dipilih dari {len(all_records)} total',
                 color='white', fontsize=12)
    ax.legend(facecolor='#2a2a4e', labelcolor='white', fontsize=9)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Map saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default=DEFAULT_SOURCE,
                        help='Path ke folder route (berisi meta/, camera/, lidar/)')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT,
                        help='Folder output untuk selected_frames.txt dan map')
    parser.add_argument('--target',    type=int,   default=DEFAULT_TARGET,
                        help='Jumlah frame yang ingin dipilih (default 500)')
    parser.add_argument('--min_vel',   type=float, default=DEFAULT_MIN_VEL,
                        help='Threshold velocity minimum m/s (default 0.5)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nUGM Frame Selector")
    print(f"  Source : {source_dir}")
    print(f"  Target : {args.target} frames")
    print(f"  Min vel: {args.min_vel} m/s\n")

    # ── 1. Load meta ──
    records = load_meta(source_dir)

    # ── 2. Select ──
    selected, total_dist = distance_based_select(
        records, min_vel=args.min_vel, target=args.target
    )

    # ── 3. Stats ──
    vels_sel = [r['vel'] for r in selected]
    print(f"\n  ── Hasil Seleksi ──")
    print(f"  Frame dipilih : {len(selected)}")
    print(f"  Interval rata : {total_dist/len(selected):.1f} m/frame")
    print(f"  Vel terpilih  : min={min(vels_sel):.2f}, max={max(vels_sel):.2f}, "
          f"avg={sum(vels_sel)/len(vels_sel):.2f} m/s")
    print(f"  Timestamp awal: {selected[0]['frame_id']}")
    print(f"  Timestamp akhir: {selected[-1]['frame_id']}")

    # ── 4. Verify files exist ──
    missing_rgb = 0
    missing_lidar = 0
    for r in selected:
        if not (source_dir / 'camera' / 'rgb' / f"{r['frame_id']}.png").exists():
            missing_rgb += 1
        if not (source_dir / 'lidar' / 'cld' / f"{r['frame_id']}.pcd").exists():
            missing_lidar += 1
    if missing_rgb or missing_lidar:
        print(f"\n  WARNING: {missing_rgb} RGB missing, {missing_lidar} LiDAR missing")
    else:
        print(f"  File check: semua RGB dan LiDAR tersedia ✓")

    # ── 5. Save frame list ──
    txt_path = output_dir / 'selected_frames.txt'
    with open(txt_path, 'w') as f:
        for r in selected:
            f.write(r['frame_id'] + '\n')
    print(f"\n  Frame list: {txt_path}")

    # ── 6. Save GPS map ──
    map_path = output_dir / 'selection_map.png'
    plot_selection(records, selected, map_path, min_vel=args.min_vel)

    print(f"\nDone. {len(selected)} frame siap untuk tahap convert & labeling.")
    print(f"  Next step: jalankan convert_ugm_to_kitti.py untuk konversi ke format KITTI")
