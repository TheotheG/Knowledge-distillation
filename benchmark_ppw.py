#!/usr/bin/env python3
"""
benchmark_ppw.py
---------------------------------------------------------------
Perf-per-watt characterization for a single TensorRT engine on Jetson.

Design (decoupled, same spirit as dump_preds.py -> one JSON per engine):
  * Latency / throughput come from `trtexec` (reference tool, already used
    to build the engines -> guaranteed to work; no pycuda / TRT-python IO).
  * Power comes from `tegrastats` sampled in parallel over the timed window.
  * Energy-per-inference and perf-per-watt are derived from the two.

The harness does NOT build engines. Feed it a prebuilt .engine.

Metrics per (engine, batch):
  latency_ms {min, mean, median, p90, p95, p99}   (end-to-end host latency)
  gpu_compute_ms {mean, median, p99}
  throughput_img_per_s
  power_mw {total_mean, per_rail_mean, n_samples, idle_total_mean}
  energy_per_image_mj
  perf_per_watt_img_per_s_per_w
  ram_used_mb {peak, baseline, delta}
Plus engine-level: engine, tag, precision, engine_load_s, trt notes, timestamp.

Usage (core perf-per-watt, batch=1, all engines):
    for e in engines/*; do
        [ -f "$e" ] || continue
        tag=$(basename "$e"); tag="${tag%.*}"
        python3 benchmark_ppw.py --engine "$e" --tag "$tag" \
            --batches 1 --out "results/ppw_${tag}.json"
    done

Batch sweep (needs an engine built with a dynamic shape profile,
i.e. --minShapes/--optShapes/--maxShapes at build time):
    python3 benchmark_ppw.py --engine engines/logitKD_fp16.engine \
        --tag logitKD_fp16 --batches 1,2,4,8,16 --out results/ppw_sweep_logitKD_fp16.json
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime

# ----- tegrastats parsing (device-specific rail names handled generically) -----
# Matches e.g. "VDD_IN 4321mW/4321mW", "VDD_CPU_GPU_CV 800mW/800mW"
RAIL_RE = re.compile(r"\b([A-Z][A-Z0-9_]+)\s+(\d+)mW/(\d+)mW")
RAM_RE = re.compile(r"\bRAM\s+(\d+)/(\d+)MB")


class TegraSampler:
    """Runs `tegrastats` in the background and timestamps every sample."""

    def __init__(self, interval_ms=100, cmd="tegrastats"):
        self.interval_ms = interval_ms
        self.cmd = cmd
        self.proc = None
        self.thread = None
        self.samples = []  # list of (t_perf_counter, {rail: mW}, ram_used_mb)
        self._stop = False

    def _reader(self):
        for line in self.proc.stdout:
            t = time.perf_counter()
            rails = {name: int(inst) for name, inst, _avg in RAIL_RE.findall(line)}
            ram = RAM_RE.search(line)
            ram_used = int(ram.group(1)) if ram else None
            if rails or ram_used is not None:
                self.samples.append((t, rails, ram_used))
            if self._stop:
                break

    def start(self):
        # Clear any straggler tegrastats instance (only one may run at a time).
        subprocess.run([self.cmd, "--stop"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        self.proc = subprocess.Popen(
            [self.cmd, "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop = True
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.thread:
            self.thread.join(timeout=3)

    def window(self, t0, t1):
        """Samples with timestamp in [t0, t1]."""
        return [s for s in self.samples if t0 <= s[0] <= t1]

    @staticmethod
    def summarize_power(win):
        """Mean per-rail mW over the window; total = VDD_IN if present else sum
        of the largest set of core rails."""
        if not win:
            return {"total_mean_mw": None, "per_rail_mean_mw": {}, "n_samples": 0}
        rail_names = set()
        for _, rails, _ in win:
            rail_names.update(rails.keys())
        per_rail = {}
        for name in rail_names:
            vals = [rails[name] for _, rails, _ in win if name in rails]
            if vals:
                per_rail[name] = sum(vals) / len(vals)
        if "VDD_IN" in per_rail:
            total = per_rail["VDD_IN"]
        else:
            # Fallback: sum every rail except VDD_IN (avoid double counting).
            total = sum(v for k, v in per_rail.items() if k != "VDD_IN")
        return {"total_mean_mw": total, "per_rail_mean_mw": per_rail,
                "n_samples": len(win)}

    @staticmethod
    def ram_stats(win):
        used = [r for _, _, r in win if r is not None]
        if not used:
            return {"peak_mb": None, "baseline_mb": None, "delta_mb": None}
        return {"peak_mb": max(used), "baseline_mb": used[0],
                "delta_mb": max(used) - used[0]}


# ----- trtexec output parsing -----
def _f(pat, text, group=1):
    m = re.search(pat, text)
    return float(m.group(group)) if m else None


def _metric_line(label_regex, text):
    """Return the tail of a trtexec metric line, tolerating the timestamp + [I]
    prefix. Anchoring on the ']' of '[I]' also stops 'Latency' from matching the
    'H2D Latency' / 'D2H Latency' lines."""
    m = re.search(r"\]\s*" + label_regex + r":\s*(.+)", text)
    if m:
        return m.group(1)
    m = re.search(r"(?:^|\s)" + label_regex + r":\s*(.+)", text, re.M)  # fallback
    return m.group(1) if m else ""


def parse_trtexec(text):
    """Extract throughput (qps) and latency/compute percentiles from trtexec log."""
    out = {}
    out["throughput_qps"] = _f(r"Throughput:\s*([\d.]+)\s*qps", text)

    lat = _metric_line(r"Latency", text)  # end-to-end host latency
    if lat:
        out["latency_ms"] = {
            "min": _f(r"min\s*=\s*([\d.]+)", lat),
            "max": _f(r"max\s*=\s*([\d.]+)", lat),
            "mean": _f(r"mean\s*=\s*([\d.]+)", lat),
            "median": _f(r"median\s*=\s*([\d.]+)", lat),
            "p90": _f(r"percentile\(90%\)\s*=\s*([\d.]+)", lat),
            "p95": _f(r"percentile\(95%\)\s*=\s*([\d.]+)", lat),
            "p99": _f(r"percentile\(99%\)\s*=\s*([\d.]+)", lat),
        }
    gpu = _metric_line(r"GPU Compute Time", text)
    if gpu:
        out["gpu_compute_ms"] = {
            "mean": _f(r"mean\s*=\s*([\d.]+)", gpu),
            "median": _f(r"median\s*=\s*([\d.]+)", gpu),
            "p99": _f(r"percentile\(99%\)\s*=\s*([\d.]+)", gpu),
        }
    out["engine_load_s"] = _f(r"Engine deserialized in\s*([\d.]+)\s*sec", text)
    return out


def run_trtexec(trtexec, engine, batch, warmup_ms, duration_s, extra=""):
    cmd = (
        f'{trtexec} --loadEngine={engine} '
        f'--shapes=input:{batch}x3x224x224 '
        f'--warmUp={warmup_ms} --duration={duration_s} --avgRuns=100 {extra}'
    )
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.stdout + "\n" + p.stderr, p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--precision", default=None,
                    help="fp32/fp16/int8 (optional label; inferred from tag if omitted)")
    ap.add_argument("--batches", default="1", help="comma list, e.g. 1,2,4,8,16")
    ap.add_argument("--warmup-ms", type=int, default=2000)
    ap.add_argument("--duration", type=float, default=15.0,
                    help="seconds of timed inference per batch")
    ap.add_argument("--steady-skip", type=float, default=3.0,
                    help="seconds skipped at window start when averaging power")
    ap.add_argument("--tegrastats-interval", type=int, default=100, help="ms")
    ap.add_argument("--trtexec", default="trtexec")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Preflight
    trtexec = shutil.which(args.trtexec) or args.trtexec
    if not (os.path.exists(trtexec) or shutil.which(args.trtexec)):
        sys.exit(f"[!] trtexec not found ({args.trtexec}). Add it to PATH.")
    if shutil.which("tegrastats") is None:
        sys.exit("[!] tegrastats not found on PATH (expected on Jetson).")
    if not os.path.exists(args.engine):
        sys.exit(f"[!] engine not found: {args.engine}")

    precision = args.precision
    if precision is None:
        for p in ("int8", "fp16", "fp32"):
            if p in args.tag.lower():
                precision = p
                break

    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    out_path = args.out or f"ppw_{args.tag}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    result = {
        "tag": args.tag,
        "engine": os.path.abspath(args.engine),
        "precision": precision,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "warmup_ms": args.warmup_ms, "duration_s": args.duration,
            "steady_skip_s": args.steady_skip,
            "tegrastats_interval_ms": args.tegrastats_interval,
        },
        "input_tensor": "input", "output_tensor": "logits",
        "by_batch": {},
    }

    sampler = TegraSampler(args.tegrastats_interval)
    sampler.start()
    time.sleep(2.0)  # capture idle baseline
    idle_win = sampler.window(0.0, time.perf_counter())
    idle_power = TegraSampler.summarize_power(idle_win).get("total_mean_mw")

    try:
        for B in batches:
            print(f"[*] {args.tag}  batch={B} ...", flush=True)
            t0 = time.perf_counter()
            log, rc = run_trtexec(trtexec, args.engine, B,
                                  args.warmup_ms, args.duration)
            t1 = time.perf_counter()

            if rc != 0 or "Throughput:" not in log:
                # Most common cause: shape outside the engine's profile.
                bad_shape = "profile" in log.lower() or "dimension" in log.lower()
                result["by_batch"][str(B)] = {
                    "error": "trtexec failed",
                    "hint": ("batch outside engine shape profile; rebuild engine "
                             "with --minShapes/--optShapes/--maxShapes"
                             if bad_shape else "see trtexec log tail"),
                    "trtexec_tail": log.strip().splitlines()[-8:],
                }
                print(f"    [!] batch={B} failed (rc={rc})", flush=True)
                continue

            m = parse_trtexec(log)
            qps = m.get("throughput_qps")
            thr_img = qps * B if qps else None

            # power over steady-state slice of the timed window
            win = sampler.window(t0 + args.steady_skip, t1)
            pw = TegraSampler.summarize_power(win)
            ram = TegraSampler.ram_stats(win)
            pw["idle_total_mean_mw"] = idle_power

            energy_mj = ppw = None
            if thr_img and pw["total_mean_mw"]:
                p_w = pw["total_mean_mw"] / 1000.0
                energy_mj = (p_w / thr_img) * 1000.0   # mJ per image
                ppw = thr_img / p_w                     # images / s / W

            result["by_batch"][str(B)] = {
                "batch": B,
                "throughput_img_per_s": thr_img,
                "throughput_qps": qps,
                "latency_ms": m.get("latency_ms"),
                "gpu_compute_ms": m.get("gpu_compute_ms"),
                "power_mw": pw,
                "energy_per_image_mj": energy_mj,
                "perf_per_watt_img_per_s_per_w": ppw,
                "ram_used_mb": ram,
            }
            if m.get("engine_load_s") and "engine_load_s" not in result:
                result["engine_load_s"] = m["engine_load_s"]

            tp = pw["total_mean_mw"]
            print(f"    thr={thr_img:.1f} img/s  "
                  f"lat_mean={m.get('latency_ms',{}).get('mean')} ms  "
                  f"power={tp/1000.0 if tp else '?'} W  "
                  f"ppw={ppw:.2f} img/s/W" if ppw else "    (power unavailable)",
                  flush=True)
    finally:
        sampler.stop()

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[OK] wrote {out_path}")

    if idle_power is None:
        print("[!] No power rails parsed from tegrastats. Paste 3s of "
              "`tegrastats` output so the rail regex can be adjusted.")


if __name__ == "__main__":
    main()
