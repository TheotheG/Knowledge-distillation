#!/usr/bin/env python3
"""
mem_coldstart.py — ISOLATED cold-start + peak memory ανά TensorRT engine.
---------------------------------------------------------------
Κάθε engine μετριέται σε ΞΕΧΩΡΙΣΤΟ subprocess (fresh CUDA/TRT context) ώστε το
"launch -> first inference ready" να ΜΗΝ μολύνεται από προηγούμενο engine (στο ppw
run το baseline_fp16 έδειξε 2.658s = first-engine CUDA init, τα υπόλοιπα ~0.18s =
warm-context reload). Εδώ ΚΑΘΕ engine πληρώνει το πραγματικό cold context init.

Modes:
  --worker : το παιδί. fresh process -> import (torch best-effort/Legion, numpy, PIL,
             polygraphy) -> 1 ΠΡΑΓΜΑΤΙΚΗ εικόνα (πρώτη valid) -> TrtRunner.activate()
             (deserialize+context) -> 1 inference -> τυπώνει JSON breakdown στο stdout.
  orchestr : για κάθε engine -> background mem sampler -> Popen worker -> wall Popen->exit
             (headline "launch->ready") + child breakdown -> median πάνω σε --repeats.

Memory (device-specific, ΜΗ συγκρίσιμα cross-device — ρητά σε caption):
  Legion : nvidia-smi memory.used (GPU package)   -> peak - baseline
  Jetson : tegrastats RAM (system unified)         -> peak - baseline
  + proc_peak_rss_mb (VmHWM child) = process-level peak, informative σε ΚΑΙ τα δύο.

Output: raw/mem/mem_<device>.json + raw/mem/mem_<device>.csv (assets μετά το validation).

Παράδειγμα (ίδιο command Legion & Jetson — auto-fallback featHardKD naming):
  python mem_coldstart.py --device jetson --engine-dir engines \
      --variants baseline logitKD featureKD featureHardKD \
      --precisions fp32 fp16 int8 --repeats 3 --warmup 1 \
      --data-root /home/user/CXpertData
"""
import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def resolve(data_root, p):
    return os.path.join(data_root, str(p).replace("CheXpert-v1.0-small/", ""))


def first_valid_path(data_root, valid_csv):
    valid_csv = valid_csv or os.path.join(data_root, "valid.csv")
    with open(valid_csv, newline="") as f:
        return next(csv.DictReader(f))["Path"]


def find_engine(engine_dir, variant, prec):
    """{dir}/{variant}_{prec}.engine, με auto-fallback στο Legion abbreviated naming."""
    cand = os.path.join(engine_dir, f"{variant}_{prec}.engine")
    if os.path.exists(cand):
        return cand
    alt = variant.replace("featureHardKD", "featHardKD")
    cand2 = os.path.join(engine_dir, f"{alt}_{prec}.engine")
    if os.path.exists(cand2):
        return cand2
    raise SystemExit(f"[!] engine not found: {cand}\n    (also tried {cand2})")


def _peak_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024.0   # kB -> MB
    except Exception:
        return None
    return None


# ============================ WORKER (child) ============================
def run_worker(args):
    t_start = time.perf_counter()
    try:
        import torch  # noqa: F401  (Legion: προφορτώνει libcudart· Jetson: αβλαβές)
    except Exception:
        pass
    import numpy as np
    from PIL import Image
    from polygraphy.backend.common import BytesFromPath
    from polygraphy.backend.trt import EngineFromBytes, TrtRunner
    t_import = time.perf_counter()

    p = first_valid_path(args.data_root, args.valid_csv)
    im = Image.open(resolve(args.data_root, p)).convert("RGB").resize((224, 224), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    x = np.ascontiguousarray(np.transpose(a, (2, 0, 1))[None])   # [1,3,224,224]
    t_prep = time.perf_counter()

    runner = TrtRunner(EngineFromBytes(BytesFromPath(args.engine)))
    runner.activate()                       # deserialize + execution context
    t_act = time.perf_counter()
    out = runner.infer({"input": x})        # first inference
    _ = out["logits"]
    t_first = time.perf_counter()
    runner.deactivate()

    rec = {
        "import_s": t_import - t_start,
        "data_prep_s": t_prep - t_import,
        "activate_s": t_act - t_prep,
        "first_infer_s": t_first - t_act,
        "ready_s": t_first - t_prep,        # engine load -> first infer (excl import/prep)
        "worker_total_s": t_first - t_start,
        "proc_peak_rss_mb": _peak_rss_mb(),
    }
    print("__MEM_JSON__ " + json.dumps(rec), flush=True)


# ============================ MEM SAMPLER (parent) ============================
class MemSampler:
    def __init__(self, device, interval_ms=100):
        self.device = device
        self.interval_ms = interval_ms
        self.samples = []          # (perf_counter, used_mb)
        self._stop = False
        self.proc = None
        self.thread = None
        self.ok = False

    def _cmd(self):
        if self.device == "legion":
            return ["nvidia-smi", "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits", "-lms", str(self.interval_ms)]
        return ["tegrastats", "--interval", str(self.interval_ms)]

    def _parse(self, line):
        if self.device == "legion":
            try:
                return float(line.strip())
            except ValueError:
                return None
        m = re.search(r"RAM (\d+)/\d+MB", line)
        return float(m.group(1)) if m else None

    def _reader(self):
        for line in self.proc.stdout:
            v = self._parse(line)
            if v is not None:
                self.samples.append((time.perf_counter(), v))
                self.ok = True
            if self._stop:
                break

    def start(self):
        self.proc = subprocess.Popen(self._cmd(), stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
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

    def baseline_before(self, t):
        prev = [v for (ts, v) in self.samples if ts <= t]
        if prev:
            return prev[-1]
        return self.samples[0][1] if self.samples else None

    def peak_in(self, t0, t1):
        vals = [v for (ts, v) in self.samples if t0 <= ts <= t1]
        return max(vals) if vals else None


# ============================ ORCHESTRATOR (parent) ============================
def summarize(vals):
    a = [v for v in vals if v is not None]
    if not a:
        return {"median": None, "min": None, "max": None, "mean": None, "n": 0, "all": []}
    return {"median": float(statistics.median(a)), "min": float(min(a)),
            "max": float(max(a)), "mean": float(statistics.fmean(a)),
            "n": len(a), "all": [round(float(v), 4) for v in a]}


def run_one(engine_path, args):
    cmd = [sys.executable, os.path.abspath(__file__), "--worker",
           "--engine", engine_path, "--data-root", args.data_root]
    if args.valid_csv:
        cmd += ["--valid-csv", args.valid_csv]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    rec = None
    for line in out.splitlines():
        if "__MEM_JSON__" in line:
            rec = json.loads(line.split("__MEM_JSON__", 1)[1].strip())
    return proc.returncode, rec, err


def orchestrate(args):
    if not args.device:
        sys.exit("[!] --device required (legion / jetson)")
    engines = [(v, p, find_engine(args.engine_dir, v, p))
               for v in args.variants for p in args.precisions]
    os.makedirs(args.out_dir, exist_ok=True)
    metric = ("nvidia-smi memory.used (GPU package, MB)" if args.device == "legion"
              else "tegrastats RAM (system unified, MB)")
    print(f"[{args.device}] engines={len(engines)} | repeats={args.repeats} | "
          f"warmup={args.warmup} | mem={metric}\n")

    sampler = MemSampler(args.device, args.interval)
    sampler.start()
    time.sleep(1.0)
    if not sampler.ok:
        hint = ("έλεγξε nvidia-smi" if args.device == "legion"
                else "μήπως τρέχει ήδη tegrastats; -> pkill tegrastats")
        print(f"[!] ΠΡΟΣΟΧΗ: mem sampler χωρίς samples ({hint})\n")

    if args.warmup > 0 and engines:
        v0, p0, path0 = engines[0]
        print(f"[warmup] {args.warmup}x throwaway στο {v0}/{p0} (ζέσταμα driver+page-cache)...")
        for _ in range(args.warmup):
            run_one(path0, args)
        print()

    results = []
    for (v, p, path) in engines:
        walls, imps, preps, acts, firsts, readys = [], [], [], [], [], []
        peaks, bases, deltas, rss = [], [], [], []
        for r in range(args.repeats):
            time.sleep(0.3)
            t_launch = time.perf_counter()
            baseline = sampler.baseline_before(t_launch)
            rc, rec, err = run_one(path, args)
            t_done = time.perf_counter()
            peak = sampler.peak_in(t_launch, t_done)
            if rc != 0 or rec is None:
                print(f"    [!] {v}/{p} rep{r} FAILED rc={rc}\n    {err.strip()[:400]}")
                continue
            walls.append(t_done - t_launch)
            imps.append(rec["import_s"]); preps.append(rec["data_prep_s"])
            acts.append(rec["activate_s"]); firsts.append(rec["first_infer_s"])
            readys.append(rec["ready_s"]); rss.append(rec.get("proc_peak_rss_mb"))
            if peak is not None and baseline is not None:
                peaks.append(peak); bases.append(baseline); deltas.append(peak - baseline)

        e = {"tag": f"{v}_{p}", "variant": v, "precision": p,
             "engine": os.path.abspath(path),
             "cold_start_wall_s": summarize(walls), "import_s": summarize(imps),
             "data_prep_s": summarize(preps), "activate_s": summarize(acts),
             "first_infer_s": summarize(firsts), "ready_s": summarize(readys),
             "peak_mem_mb": summarize(peaks), "baseline_mem_mb": summarize(bases),
             "delta_mem_mb": summarize(deltas), "proc_peak_rss_mb": summarize(rss)}
        results.append(e)

        def fs(x): return "n/a" if x is None else f"{x:.3f}"
        def fm(x): return "n/a" if x is None else f"{x:.0f}"
        print(f"[{v}/{p}]  wall={fs(e['cold_start_wall_s']['median'])}s  "
              f"ready={fs(e['ready_s']['median'])}s "
              f"(act={fs(e['activate_s']['median'])} infer={fs(e['first_infer_s']['median'])})  "
              f"peak={fm(e['peak_mem_mb']['median'])}MB "
              f"Δ={fm(e['delta_mem_mb']['median'])}MB  rss={fm(e['proc_peak_rss_mb']['median'])}MB")

    sampler.stop()

    payload = {"device": args.device, "mem_metric": metric, "repeats": args.repeats,
               "warmup_runs": args.warmup, "sampler_interval_ms": args.interval,
               "note": "cross-device memory NON-comparable (Legion GPU-mem vs Jetson system-RAM)",
               "timestamp": datetime.now().isoformat(timespec="seconds"),
               "engines": results}
    jpath = os.path.join(args.out_dir, f"mem_{args.device}.json")
    with open(jpath, "w") as f:
        json.dump(payload, f, indent=2)

    def g(e, k):
        v = e[k]["median"]
        return f"{v:.4f}" if v is not None else ""
    cpath = os.path.join(args.out_dir, f"mem_{args.device}.csv")
    with open(cpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["device", "variant", "precision", "cold_start_wall_s", "import_s",
                    "activate_s", "first_infer_s", "ready_s",
                    "peak_mem_mb", "delta_mem_mb", "proc_peak_rss_mb"])
        for e in results:
            w.writerow([args.device, e["variant"], e["precision"],
                        g(e, "cold_start_wall_s"), g(e, "import_s"), g(e, "activate_s"),
                        g(e, "first_infer_s"), g(e, "ready_s"),
                        g(e, "peak_mem_mb"), g(e, "delta_mem_mb"), g(e, "proc_peak_rss_mb")])
    print(f"\n[json] {jpath}\n[csv]  {cpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--engine", default=None)                 # worker
    ap.add_argument("--device", default=None)                 # legion / jetson
    ap.add_argument("--engine-dir", default="engines")
    ap.add_argument("--variants", nargs="+",
                    default=["baseline", "logitKD", "featureKD", "featureHardKD"])
    ap.add_argument("--precisions", nargs="+", default=["fp32", "fp16", "int8"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--interval", type=int, default=100)      # sampler ms
    ap.add_argument("--data-root", default="/home/user/CXpertData")
    ap.add_argument("--valid-csv", default=None)
    ap.add_argument("--out-dir", default="raw/mem")
    args = ap.parse_args()

    if args.worker:
        if not args.engine:
            sys.exit("[!] --worker needs --engine")
        run_worker(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()