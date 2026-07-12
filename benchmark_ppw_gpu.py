#!/usr/bin/env python3
"""
benchmark_ppw_gpu.py  (v2 — trtexec-free)
---------------------------------------------------------------
Perf-per-watt characterization για ΕΝΑ TensorRT engine σε DESKTOP/LAPTOP NVIDIA
GPU (Legion). Sibling του benchmark_ppw.py (Jetson). ΙΔΙΟ JSON schema.

Γιατί όχι trtexec: το pip `tensorrt` wheel ΔΕΝ περιλαμβάνει trtexec (υπάρχει μόνο
στο tar/deb TensorRT). Στο Legion τα engines τρέχουν μέσω TRT Python runtime
(όπως το eval_engines.py). Οπότε εδώ:

  * Inference : raw TensorRT Python (deserialize + execute_async_v3) σε torch stream
  * Latency   : CUDA events (torch) -> gpu_compute_ms percentiles (== trtexec metric)
  * Throughput: continuous timed loop χωρίς per-call sync (saturated, όπως trtexec)
  * Power/mem/temp : nvidia-smi (NvsmiSampler)

CAVEATS (paper): power_domain=gpu_package_only vs Jetson board power· harness
Legion=raw-TRT/CUDA-events vs Jetson=trtexec -> συγκρίνουμε throughput & gpu_compute.

ΠΡΩΤΟ ΤΡΕΞΙΜΟ: `--self-test` σε ένα engine (1 inference, τυπώνει I/O + shapes).
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

import numpy as np


# ============================ nvidia-smi sampler ============================
class NvsmiSampler:
    QUERY = "power.draw,memory.used,temperature.gpu,clocks.current.sm"

    def __init__(self, interval_ms=100, gpu_index=0, nvsmi="nvidia-smi"):
        self.interval_ms = interval_ms
        self.gpu_index = gpu_index
        self.nvsmi = nvsmi
        self.proc = None
        self.thread = None
        self.samples = []
        self._stop = False

    @staticmethod
    def _num(tok):
        tok = tok.strip()
        if not tok or tok.upper().startswith("[N/A") or tok.upper() == "N/A":
            return None
        try:
            return float(tok)
        except ValueError:
            return None

    def _reader(self):
        for line in self.proc.stdout:
            t = time.perf_counter()
            parts = line.split(",")
            if len(parts) < 4:
                continue
            pw, mem, temp, sm = (self._num(parts[0]), self._num(parts[1]),
                                 self._num(parts[2]), self._num(parts[3]))
            self.samples.append((t, pw * 1000.0 if pw is not None else None, mem, temp, sm))
            if self._stop:
                break

    def start(self):
        cmd = [self.nvsmi, f"--query-gpu={self.QUERY}",
               "--format=csv,noheader,nounits",
               "-i", str(self.gpu_index), "-lms", str(self.interval_ms)]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
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
        if self.thread:
            self.thread.join(timeout=3)

    def window(self, t0, t1):
        return [s for s in self.samples if t0 <= s[0] <= t1]

    @staticmethod
    def summarize_power(win):
        vals = [s[1] for s in win if s[1] is not None]
        if not vals:
            return {"total_mean_mw": None, "per_rail_mean_mw": {}, "n_samples": len(win)}
        mean = sum(vals) / len(vals)
        return {"total_mean_mw": mean, "per_rail_mean_mw": {"GPU": mean},
                "n_samples": len(vals)}

    @staticmethod
    def ram_stats(win):
        used = [s[2] for s in win if s[2] is not None]
        if not used:
            return {"peak_mb": None, "baseline_mb": None, "delta_mb": None}
        return {"peak_mb": max(used), "baseline_mb": used[0],
                "delta_mb": max(used) - used[0]}

    @staticmethod
    def thermal_stats(win):
        temps = [s[3] for s in win if s[3] is not None]
        clks = [s[4] for s in win if s[4] is not None]
        out = {}
        if temps:
            out.update(temp_mean_c=sum(temps) / len(temps), temp_max_c=max(temps))
        if clks:
            out.update(sm_clock_mean_mhz=sum(clks) / len(clks), sm_clock_min_mhz=min(clks))
        return out or None


# ============================ TensorRT runner (raw) ============================
def _preload_cuda():
    import torch  # noqa
    return torch


def _trt_dtype_to_torch(trt, dt):
    import torch
    m = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.BOOL: torch.bool,
    }
    return m.get(dt, torch.float32)


class TrtRunner:
    def __init__(self, engine_path, verbose=False):
        self.torch = _preload_cuda()
        import tensorrt as trt
        self.trt = trt
        self.trt_version = trt.__version__
        logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            blob = f.read()
        t0 = time.perf_counter()
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(blob)
        self.engine_load_s = time.perf_counter() - t0
        if self.engine is None:
            raise RuntimeError("deserialize_cuda_engine returned None "
                               "(TRT version mismatch vs build).")
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
            else:
                self.outputs.append(name)
        if not self.inputs or not self.outputs:
            raise RuntimeError(f"unexpected IO: in={self.inputs} out={self.outputs}")
        self.in_name = self.inputs[0]
        self.stream = self.torch.cuda.Stream()
        self.bufs = {}

    def setup(self, batch, hw=(3, 224, 224)):
        trt, torch = self.trt, self.torch
        shp = (batch, *hw)
        self.context.set_input_shape(self.in_name, shp)
        if not self.context.all_binding_shapes_specified:
            raise RuntimeError("binding shapes not fully specified")
        self.bufs = {}
        in_dt = _trt_dtype_to_torch(trt, self.engine.get_tensor_dtype(self.in_name))
        self.bufs[self.in_name] = torch.randn(
            shp, device="cuda", dtype=torch.float32).to(in_dt).contiguous()
        for name in self.outputs:
            oshape = tuple(self.context.get_tensor_shape(name))
            odt = _trt_dtype_to_torch(trt, self.engine.get_tensor_dtype(name))
            self.bufs[name] = torch.empty(oshape, device="cuda", dtype=odt).contiguous()
        for name, buf in self.bufs.items():
            self.context.set_tensor_address(name, buf.data_ptr())

    def _infer(self):
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("execute_async_v3 returned False")

    def io_report(self):
        return {
            "trt_version": self.trt_version,
            "inputs": [(n, str(self.engine.get_tensor_dtype(n))) for n in self.inputs],
            "outputs": [(n, tuple(self.context.get_tensor_shape(n)),
                         str(self.engine.get_tensor_dtype(n))) for n in self.outputs],
        }

    def throughput_window(self, batch, warmup_ms, duration_s):
        torch = self.torch
        self.setup(batch)
        with torch.cuda.stream(self.stream):
            tw = time.perf_counter()
            while (time.perf_counter() - tw) * 1000.0 < warmup_ms:
                self._infer()
            self.stream.synchronize()
            iters = 0
            t0 = time.perf_counter()
            while (time.perf_counter() - t0) < duration_s:
                self._infer()
                iters += 1
            self.stream.synchronize()
            t1 = time.perf_counter()
        thr_img = (iters * batch) / (t1 - t0)
        return thr_img, iters, t0, t1

    def latency_pass(self, batch, n_iters=200):
        torch = self.torch
        gpu_ms, e2e_ms = [], []
        with torch.cuda.stream(self.stream):
            for _ in range(min(30, n_iters)):
                self._infer()
            self.stream.synchronize()
            for _ in range(n_iters):
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                w0 = time.perf_counter()
                ev0.record(self.stream)
                self._infer()
                ev1.record(self.stream)
                self.stream.synchronize()
                w1 = time.perf_counter()
                gpu_ms.append(ev0.elapsed_time(ev1))
                e2e_ms.append((w1 - w0) * 1000.0)
        return np.array(gpu_ms), np.array(e2e_ms)


def _pct(a):
    a = np.asarray(a, dtype=np.float64)
    return {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()),
            "median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99))}


# ============================ main ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--precision", default=None)
    ap.add_argument("--batches", default="1")
    ap.add_argument("--warmup-ms", type=int, default=2000)
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--steady-skip", type=float, default=3.0)
    ap.add_argument("--lat-iters", type=int, default=200)
    ap.add_argument("--nvsmi-interval", type=int, default=100)
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--nvsmi", default="nvidia-smi")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.engine):
        sys.exit(f"[!] engine not found: {args.engine}")

    try:
        runner = TrtRunner(args.engine, verbose=args.self_test)
    except Exception as e:
        print("[!] Αποτυχία φόρτωσης engine μέσω TensorRT Python. Diagnostics:")
        print(f"    engine : {args.engine}")
        print(f"    error  : {type(e).__name__}: {e}")
        print("    check  : python -c \"import tensorrt,torch;print(tensorrt.__version__, torch.version.cuda)\"")
        sys.exit(1)

    if args.self_test:
        runner.throughput_window(1, warmup_ms=200, duration_s=0.5)
        rep = runner.io_report()
        print("[self-test OK]")
        print(f"    trt_version : {rep['trt_version']}")
        print(f"    engine_load : {runner.engine_load_s:.3f} s")
        print(f"    inputs      : {rep['inputs']}")
        print(f"    outputs     : {rep['outputs']}")
        return

    precision = args.precision
    if precision is None:
        for p in ("int8", "fp16", "fp32"):
            if p in args.tag.lower():
                precision = p
                break

    import shutil
    if shutil.which(args.nvsmi) is None:
        sys.exit("[!] nvidia-smi not found on PATH.")
    chk = subprocess.run([args.nvsmi, "--query-gpu=power.draw",
                          "--format=csv,noheader,nounits", "-i", str(args.gpu_index)],
                         capture_output=True, text=True)
    if "N/A" in chk.stdout.upper() or chk.returncode != 0:
        print("[!] ΠΡΟΣΟΧΗ: power.draw = N/A -> perf-per-watt δεν θα υπολογιστεί.")

    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    out_path = args.out or f"ppw_{args.tag}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    result = {
        "tag": args.tag, "engine": os.path.abspath(args.engine),
        "precision": precision, "platform": "legion_gpu",
        "power_domain": "gpu_package_only", "power_source": "nvidia-smi power.draw",
        "harness": "raw_trt_cuda_events", "trt_version": runner.trt_version,
        "engine_load_s": runner.engine_load_s,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {"warmup_ms": args.warmup_ms, "duration_s": args.duration,
                   "steady_skip_s": args.steady_skip, "lat_iters": args.lat_iters,
                   "nvsmi_interval_ms": args.nvsmi_interval, "gpu_index": args.gpu_index},
        "input_tensor": runner.in_name, "output_tensor": runner.outputs[0],
        "by_batch": {},
    }

    sampler = NvsmiSampler(args.nvsmi_interval, args.gpu_index, args.nvsmi)
    sampler.start()
    time.sleep(2.0)
    idle_power = NvsmiSampler.summarize_power(
        sampler.window(0.0, time.perf_counter())).get("total_mean_mw")

    try:
        for B in batches:
            print(f"[*] {args.tag}  batch={B} ...", flush=True)
            try:
                thr_img, iters, t0, t1 = runner.throughput_window(
                    B, args.warmup_ms, args.duration)
                gpu_ms, e2e_ms = runner.latency_pass(B, args.lat_iters)
            except Exception as e:
                result["by_batch"][str(B)] = {"error": f"{type(e).__name__}: {e}"}
                print(f"    [!] batch={B} failed: {e}", flush=True)
                continue

            win = sampler.window(t0 + args.steady_skip, t1)
            pw = NvsmiSampler.summarize_power(win)
            ram = NvsmiSampler.ram_stats(win)
            therm = NvsmiSampler.thermal_stats(win)
            pw["idle_total_mean_mw"] = idle_power

            energy_mj = ppw = None
            if thr_img and pw["total_mean_mw"]:
                p_w = pw["total_mean_mw"] / 1000.0
                energy_mj = (p_w / thr_img) * 1000.0
                ppw = thr_img / p_w

            result["by_batch"][str(B)] = {
                "batch": B, "throughput_img_per_s": thr_img,
                "throughput_qps": thr_img / B if thr_img else None,
                "measured_iters": iters,
                "latency_ms": _pct(e2e_ms), "gpu_compute_ms": _pct(gpu_ms),
                "power_mw": pw, "energy_per_image_mj": energy_mj,
                "perf_per_watt_img_per_s_per_w": ppw,
                "ram_used_mb": ram, "gpu_thermal": therm,
            }
            tp = pw["total_mean_mw"]
            if ppw:
                print(f"    thr={thr_img:.1f} img/s  gpu={_pct(gpu_ms)['mean']:.3f} ms  "
                      f"power={tp/1000.0:.2f} W  ppw={ppw:.2f} img/s/W", flush=True)
            else:
                print(f"    thr={thr_img:.1f} img/s  (power unavailable)", flush=True)
    finally:
        sampler.stop()

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
