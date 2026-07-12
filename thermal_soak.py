#!/usr/bin/env python3
"""
thermal_soak.py — A5: sustained-load thermal / throttling test.
---------------------------------------------------------------
Συνεχές inference (single-stream = real-time deployment) σε ένα engine για
--duration s, με background thread που δειγματοληπτεί temp/power/clock/util ανά ~1s.
Main thread μετράει throughput ανά --window s -> χρονοσειρά. Δείχνει αν ανεβαίνει
η θερμοκρασία, πέφτουν τα clocks (throttling) και χάνεται throughput σε steady-state.

Τηλεμετρία (torch-free, ίδιο footprint με eval_engines.py):
  legion : nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,utilization.gpu
  jetson : tegrastats --interval 1000  (gpu@..C, VDD_IN ..mW, GR3D_FREQ ..%)

ΠΡΟΣΟΧΗ (caption): power domains ΜΗ συγκρίσιμα — Legion GPU-package vs Jetson VDD_IN.

Έξοδος ανά engine: raw/thermal/thermal_<device>_<variant>_<prec>.json
"""
import argparse
import csv
import json
import os
import re
import subprocess
import threading
import time

import numpy as np
from PIL import Image

try:
    import torch  # noqa: F401  (προφορτώνει libcudart στο Legion conda env)
except Exception:
    pass

PATHOLOGIES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve(data_root, p):
    return os.path.join(data_root, str(p).replace("CheXpert-v1.0-small/", ""))


def preprocess(path):
    img = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - MEAN) / STD
    return np.transpose(a, (2, 0, 1))


def load_paths(data_root, valid_csv):
    valid_csv = valid_csv or os.path.join(data_root, "valid.csv")
    with open(valid_csv, newline="") as f:
        return [row["Path"] for row in csv.DictReader(f)]


def find_engine(engine_dir, variant, prec):
    cand = os.path.join(engine_dir, f"{variant}_{prec}.engine")
    if os.path.exists(cand):
        return cand
    alt = os.path.join(engine_dir, f"{variant.replace('featureHardKD', 'featHardKD')}_{prec}.engine")
    if os.path.exists(alt):
        return alt
    raise SystemExit(f"[!] engine not found: {cand} (also tried {alt})")


# ============================ TELEMETRY PARSERS ============================
def parse_tegrastats(line):
    """tegrastats -> (gpu_temp_c, power_w, gpu_util_pct, gpu_clock_mhz|None)."""
    temp = re.search(r"gpu@([\d.]+)C", line)
    vdd = re.search(r"VDD_IN (\d+)mW", line)
    util = re.search(r"GR3D_FREQ (\d+)%", line)
    clk = re.search(r"GR3D_FREQ \d+%@(\d+)", line)
    return (
        float(temp.group(1)) if temp else None,
        (float(vdd.group(1)) / 1000.0) if vdd else None,
        float(util.group(1)) if util else None,
        float(clk.group(1)) if clk else None,
    )


def parse_nvidia_smi(line):
    """'temp, power, clock_sm, util' -> (temp, power, util, clock)."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        return (None, None, None, None)
    def f(x):
        try:
            return float(x)
        except ValueError:
            return None
    temp, power, clk, util = f(parts[0]), f(parts[1]), f(parts[2]), f(parts[3])
    return (temp, power, util, clk)


# ============================ SAMPLERS ============================
def sample_legion(stop_evt, t0, out):
    cmd = ["nvidia-smi",
           "--query-gpu=temperature.gpu,power.draw,clocks.sm,utilization.gpu",
           "--format=csv,noheader,nounits"]
    while not stop_evt.is_set():
        try:
            line = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=5).stdout.strip().splitlines()[0]
            temp, power, util, clk = parse_nvidia_smi(line)
            out.append(dict(t=round(time.time() - t0, 2), gpu_temp_c=temp,
                            power_w=power, gpu_clock_mhz=clk, gpu_util_pct=util))
        except Exception:
            pass
        stop_evt.wait(1.0)


def sample_jetson(stop_evt, t0, out):
    proc = subprocess.Popen(["tegrastats", "--interval", "1000"],
                            stdout=subprocess.PIPE, text=True)
    try:
        for line in proc.stdout:
            if stop_evt.is_set():
                break
            temp, power, util, clk = parse_tegrastats(line)
            out.append(dict(t=round(time.time() - t0, 2), gpu_temp_c=temp,
                            power_w=power, gpu_clock_mhz=clk, gpu_util_pct=util))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


# ============================ SUMMARY ============================
def _mean(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def summarize(telemetry, throughput, duration, steady_frac=0.25, thr_pct=97.0):
    t_start = duration * 0.10
    t_steady = duration * (1.0 - steady_frac)
    tele_start = [s for s in telemetry if s["t"] <= t_start]
    tele_steady = [s for s in telemetry if s["t"] >= t_steady]
    thr_start = [w for w in throughput if w["t"] <= max(t_start, throughput[0]["t"])] if throughput else []
    thr_steady = [w for w in throughput if w["t"] >= t_steady]
    temps = [s["gpu_temp_c"] for s in telemetry if s["gpu_temp_c"] is not None]
    clk_start = _mean([s["gpu_clock_mhz"] for s in tele_start])
    clk_steady = _mean([s["gpu_clock_mhz"] for s in tele_steady])
    tp_start = _mean([w["img_s"] for w in thr_start]) if thr_start else None
    tp_steady = _mean([w["img_s"] for w in thr_steady]) if thr_steady else None
    tp_ret = (100.0 * tp_steady / tp_start) if (tp_start and tp_steady) else None
    clk_ret = (100.0 * clk_steady / clk_start) if (clk_start and clk_steady) else None
    throttled = bool((tp_ret is not None and tp_ret < thr_pct) or
                     (clk_ret is not None and clk_ret < thr_pct))
    return dict(
        temp_start_c=_mean([s["gpu_temp_c"] for s in tele_start]),
        temp_steady_c=_mean([s["gpu_temp_c"] for s in tele_steady]),
        temp_max_c=(max(temps) if temps else None),
        power_steady_w=_mean([s["power_w"] for s in tele_steady]),
        clock_start_mhz=clk_start, clock_steady_mhz=clk_steady,
        throughput_start_img_s=tp_start, throughput_steady_img_s=tp_steady,
        throughput_retention_pct=(round(tp_ret, 2) if tp_ret else None),
        clock_retention_pct=(round(clk_ret, 2) if clk_ret else None),
        throttled=throttled,
    )


# ============================ SOAK ONE ENGINE ============================
def soak(engine_path, imgs, device, duration, window):
    from polygraphy.backend.common import BytesFromPath
    from polygraphy.backend.trt import EngineFromBytes, TrtRunner
    telemetry = []
    stop_evt = threading.Event()
    t0 = time.time()
    sampler = (sample_jetson if device == "jetson" else sample_legion)
    th = threading.Thread(target=sampler, args=(stop_evt, t0, telemetry), daemon=True)
    th.start()
    throughput, win_count, win_start = [], 0, t0
    N = len(imgs)
    try:
        with TrtRunner(EngineFromBytes(BytesFromPath(engine_path))) as r:
            i = 0
            while True:
                b = np.ascontiguousarray(imgs[i % N][None])   # batch=1
                r.infer({"input": b})
                win_count += 1
                i += 1
                now = time.time()
                if now - win_start >= window:
                    throughput.append(dict(t=round(now - t0, 2),
                                           img_s=round(win_count / (now - win_start), 2)))
                    win_count, win_start = 0, now
                if now - t0 >= duration:
                    break
    finally:
        stop_evt.set()
        th.join(timeout=5)
    return telemetry, throughput


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, choices=["legion", "jetson"])
    ap.add_argument("--engine-dir", default="engines")
    ap.add_argument("--variant", default="logitKD")
    ap.add_argument("--precisions", nargs="+", default=["fp16", "int8"])
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--cooldown", type=float, default=60.0,
                    help="idle sleep ΜΕΤΑΞΥ engines ώστε το int8 να ξεκινά cool")
    ap.add_argument("--data-root", default="/home/user/CXpertData")
    ap.add_argument("--valid-csv", default=None)
    ap.add_argument("--out-dir", default="raw/thermal")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = load_paths(args.data_root, args.valid_csv)
    imgs = np.stack([preprocess(resolve(args.data_root, p)) for p in paths]).astype(np.float32)
    print(f"[{args.device}] {len(imgs)} images | variant={args.variant} | "
          f"precisions={args.precisions} | {args.duration:.0f}s each\n")

    for k, prec in enumerate(args.precisions):
        if k > 0 and args.cooldown > 0:
            print(f"  cooldown {args.cooldown:.0f}s (idle)...")
            time.sleep(args.cooldown)
        eng = find_engine(args.engine_dir, args.variant, prec)
        print(f"[soak] {args.variant}/{prec}  ({eng})  {args.duration:.0f}s ...")
        telemetry, throughput = soak(eng, imgs, args.device, args.duration, args.window)
        summ = summarize(telemetry, throughput, args.duration)
        out = os.path.join(args.out_dir,
                           f"thermal_{args.device}_{args.variant}_{prec}.json")
        with open(out, "w") as f:
            json.dump(dict(device=args.device, variant=args.variant, precision=prec,
                           duration_s=args.duration, window_s=args.window,
                           telemetry=telemetry, throughput=throughput,
                           summary=summ), f, indent=2)
        print(f"    temp: {summ['temp_start_c']}→{summ['temp_steady_c']}°C "
              f"(max {summ['temp_max_c']})  |  power steady {summ['power_steady_w']}W")
        print(f"    clock: {summ['clock_start_mhz']}→{summ['clock_steady_mhz']} MHz "
              f"(ret {summ['clock_retention_pct']}%)")
        print(f"    thrpt: {summ['throughput_start_img_s']}→"
              f"{summ['throughput_steady_img_s']} img/s "
              f"(ret {summ['throughput_retention_pct']}%)  "
              f"THROTTLED={summ['throttled']}")
        print(f"    [json] {out}\n")


if __name__ == "__main__":
    main()