#!/usr/bin/env python3
"""One configuration: fresh server, 1 warmup + N timed /v1/videos requests.

E2E is measured from POST to the first observed `completed`. Per-stage times come
from the server's own perf log (SGLANG_PERF_LOG_DIR/performance.log, one JSON
line per request). Layout per exp-convention: <config>/cmd.sh, logs/, results/.
"""
import argparse
import json
import re
import os
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from http.client import HTTPException  # the local helper is named http()
from pathlib import Path


def stats(values):
    values = [float(v) for v in values]
    med = statistics.median(values)
    mad = statistics.median([abs(v - med) for v in values])
    return {
        "count": len(values), "values": values, "median": med,
        "mean": statistics.fmean(values), "min": min(values), "max": max(values),
        "mad": mad, "relative_mad": (mad / med) if med else 0.0,
    }


def http(method, url, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read() or b"{}")


def port_free(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def cpulist(text):
    out = set()
    for part in text.strip().split(","):
        if not part:
            continue
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def numa_prefix(log):
    """Bind to the allocated CPUs; membind to the GPUs' NUMA node when unambiguous."""
    if subprocess.run(["which", "numactl"], capture_output=True).returncode != 0:
        log("numactl not found; no binding")
        return [], {"binding": None}
    cpus = set(os.sched_getaffinity(0))
    nodes = {}
    for p in Path("/sys/devices/system/node").glob("node[0-9]*"):
        nodes[int(p.name[4:])] = cpulist((p / "cpulist").read_text())
    per_node = {n: sorted(cpus & s) for n, s in nodes.items() if cpus & s}
    gpu_nodes = set()
    try:
        ids = subprocess.run(["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout.split()
        for b in ids:
            dom, bus, rest = b.split(":", 2)
            path = Path(f"/sys/bus/pci/devices/{dom[-4:].lower()}:{bus.lower()}:{rest.lower()}/numa_node")
            gpu_nodes.add(int(path.read_text()))
    except Exception as exc:  # noqa: BLE001
        log(f"gpu numa lookup failed: {exc!r}")
    info = {"allocated_cpus": sorted(cpus), "cpus_per_node": {str(k): v for k, v in per_node.items()},
            "gpu_numa_nodes": sorted(gpu_nodes)}
    if len(gpu_nodes) == 1 and per_node.get(next(iter(gpu_nodes))):
        node = next(iter(gpu_nodes))
        info["binding"] = f"physcpubind=node{node} membind={node}"
        return ["numactl", "--physcpubind=" + ",".join(map(str, per_node[node])), f"--membind={node}"], info
    if len(per_node) == 1:
        node = next(iter(per_node))
        info["binding"] = f"physcpubind=all membind={node}"
        return ["numactl", "--physcpubind=" + ",".join(map(str, sorted(cpus))), f"--membind={node}"], info
    info["binding"] = "physcpubind=all"
    return ["numactl", "--physcpubind=" + ",".join(map(str, sorted(cpus)))], info


def shift_ports(argv, base_port, offset, stride=10):
    """Rewrite --port/--master-port so every run uses ports never used before in this allocation."""
    argv = list(argv)
    new_port = base_port + stride * offset
    for flag in ("--port", "--master-port"):
        if flag in argv:
            i = argv.index(flag)
            argv[i + 1] = str(int(argv[i + 1]) + stride * offset)
    return argv, new_port


def substitute(value, mapping):
    if isinstance(value, str):
        for k, v in mapping.items():
            value = value.replace("{" + k + "}", v)
        return value
    if isinstance(value, list):
        return [substitute(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k: substitute(v, mapping) for k, v in value.items()}
    return value


STAGE_RE = re.compile(r"\[(\w+Stage)\] finished in ([0-9.]+) seconds")


def server_log_stages(server_log_path, offset):
    """Stage durations (ms) logged by the server since byte `offset`; returns (stages, new_offset)."""
    with open(server_log_path, "rb") as f:
        f.seek(offset)
        chunk = f.read()
    stages = {}
    for m in STAGE_RE.finditer(chunk.decode("utf-8", "replace")):
        stages[m.group(1)] = float(m.group(2)) * 1000.0
    return stages, offset + len(chunk)


def wait_perf_lines(path, count, timeout_s, log):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            lines = [l for l in path.read_text().splitlines() if l.strip()]
            if len(lines) >= count:
                return [json.loads(l) for l in lines]
        time.sleep(0.2)
    log(f"perf log has fewer than {count} records after {timeout_s}s")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path)
    ap.add_argument("--output", type=Path, help="new config directory, must not exist")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        s = stats([3, 1, 2, 10])
        assert s["median"] == 2.5 and s["min"] == 1 and s["max"] == 10 and s["mad"] == 1.0
        assert cpulist("0-2,5") == {0, 1, 2, 5}
        a, prt = shift_ports(["x", "--port", "50092", "--master-port", "50100"], 50092, 3)
        assert a == ["x", "--port", "50122", "--master-port", "50130"] and prt == 50122, (a, prt)
        print("selftest ok")
        return
    if not a.config or not a.output:
        raise SystemExit("--config and --output are required")
    cfg = json.loads(a.config.read_text())
    out = a.output.resolve()
    if out.exists():
        raise SystemExit(f"refusing to reuse {out}; use a -2/-3 suffix")
    for d in ("logs", "results", "results/perf", "outputs", "inputs"):
        (out / d).mkdir(parents=True)
    log_f = open(out / "logs" / "runner.log", "a", buffering=1)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")

    mapping = {"config_dir": str(out), "exp_dir": str(out.parent)}
    cfg = substitute(cfg, mapping)
    counter_file = out.parent / "logs" / "port-offset.txt"
    offset = (int(counter_file.read_text()) if counter_file.exists() else 0) + 1
    counter_file.write_text(str(offset))
    cfg["server_argv"], port = shift_ports(cfg["server_argv"], int(cfg["port"]), offset)
    cfg["port_offset"] = offset
    if not port_free(port):
        raise RuntimeError(f"Error: port {port} busy before launch (stale server from a previous run?)")
    env = dict(os.environ)
    env.update(cfg["env"])
    for k in ("TMPDIR", "TMP", "TEMP"):  # nsys and ffmpeg need an existing temp dir before the server creates one
        if env.get(k):
            Path(env[k]).mkdir(parents=True, exist_ok=True)
    env["SGLANG_PERF_LOG_DIR"] = str(out / "results" / "perf")
    prefix, numa = numa_prefix(log)
    session = f"h3vae-{int(time.time())}"
    profile = bool(cfg.get("profile"))
    nsys_prefix = ["nsys", "launch", f"--session-new={session}", "--trace=cuda,nvtx",
                   "--trace-fork-before-exec=true", "--wait=all"] if profile else []
    argv = nsys_prefix + prefix + cfg["server_argv"]
    if profile:
        (out / "profile").mkdir()
    with open(out / "cmd.sh", "w") as f:
        f.write("#!/bin/bash\n# actual server launch; request bodies in results/raw.json\n")
        f.write(f"cd {shlex.quote(str(out))}\n")
        for k, v in sorted(cfg["env"].items()):
            f.write(f"export {k}={shlex.quote(v)}\n")
        f.write(f"export SGLANG_PERF_LOG_DIR={shlex.quote(env['SGLANG_PERF_LOG_DIR'])}\n")
        f.write(" ".join(shlex.quote(x) for x in argv) + "\n")
    log(f"launch: {' '.join(shlex.quote(x) for x in argv)}")
    server_log = open(out / "logs" / "server.log", "ab")
    proc = subprocess.Popen(argv, cwd=str(out), env=env, stdout=server_log,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    raw = {"id": cfg["id"], "config": cfg, "numa": numa, "hostname": socket.gethostname(),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "requests": [], "server_pid": proc.pid}
    try:
        t_start = time.time()
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited early with {proc.returncode}; see logs/server.log")
            try:
                if http("GET", base + "/health", timeout=5)[0] == 200:
                    break
            except (urllib.error.URLError, ConnectionError, OSError, HTTPException):
                pass  # not up yet, or a transient non-HTTP socket on the port during startup
            bound = re.findall(rb"Uvicorn running on http://[0-9.]+:(\d+)", (out / "logs" / "server.log").read_bytes())
            if bound and int(bound[-1]) != port:
                raise RuntimeError(f"server bound port {bound[-1].decode()} instead of {port}; port {port} was busy (stale server?)")
            if time.time() - t_start > cfg.get("ready_timeout_s", 1800):
                raise TimeoutError("server not ready in time")
            time.sleep(2)
        raw["startup_s"] = time.time() - t_start
        log(f"server ready after {raw['startup_s']:.0f}s")
        perf_path = out / "results" / "perf" / "performance.log"
        roles = ["warmup"] * int(cfg.get("warmup", 1)) + ["timed"] * int(cfg.get("timed", 7))
        server_log_path = out / "logs" / "server.log"
        log_offset = server_log_path.stat().st_size
        for i, role in enumerate(roles):
            body = json.loads(json.dumps(cfg["request"]))
            body["output_path"] = str(out / "outputs" / f"{i:02d}-{role}")
            if cfg.get("quality"):
                body["quality"] = cfg["quality"]
            if profile and role == "timed":
                report = str(out / "profile" / f"request{i}")
                r = subprocess.run(["nsys", "start", f"--session={session}", "--sample=none", "--cpuctxsw=none",
                                    f"--output={report}", "--force-overwrite=true"], env=env, capture_output=True, text=True)
                log(f"nsys start rc={r.returncode} {r.stdout.strip()[:200]!r} {r.stderr.strip()[:200]!r}")
                if r.returncode != 0:
                    raise RuntimeError("nsys start failed")
            t0 = time.perf_counter()
            status, job = http("POST", base + "/v1/videos", body, timeout=cfg.get("request_timeout_s", 900))
            job_id = job.get("id")
            deadline = time.time() + cfg.get("request_timeout_s", 900)
            failures = 0
            while True:
                try:
                    _, j = http("GET", f"{base}/v1/videos/{job_id}", timeout=30)
                except (urllib.error.URLError, ConnectionError, OSError, HTTPException) as exc:
                    failures += 1
                    if failures > 20:
                        raise
                    log(f"status poll error ({failures}): {exc!r}")
                    time.sleep(0.5)
                    continue
                if j.get("status") == "completed":
                    t1 = time.perf_counter()
                    break
                if j.get("status") == "failed":
                    raise RuntimeError(f"request {i} failed: {j}")
                if time.time() > deadline:
                    raise TimeoutError(f"request {i} timed out")
                time.sleep(cfg.get("poll_interval_s", 0.05))
            e2e_ms = (t1 - t0) * 1000
            if profile and role == "timed":
                r = subprocess.run(["nsys", "stop", f"--session={session}"], env=env, capture_output=True, text=True)
                log(f"nsys stop rc={r.returncode} stdout={r.stdout.strip()[:300]!r} stderr={r.stderr.strip()[:300]!r}")
                for _ in range(180):  # the report is finalized asynchronously; wait for it before shutting down
                    if Path(report + ".nsys-rep").exists():
                        break
                    time.sleep(1)
                log("report present: %s" % Path(report + ".nsys-rep").exists())
            time.sleep(1.0)  # let the server flush its stage lines
            log_stages, log_offset = server_log_stages(server_log_path, log_offset)
            records = wait_perf_lines(perf_path, i + 1, 15, log)
            perf = next((r for r in records if r.get("request_id") == job_id), records[i] if len(records) > i else None)
            stages = {s["name"]: s["execution_time_ms"] for s in (perf or {}).get("stages", [])} or log_stages
            rec = {"index": i, "role": role, "job_id": job_id, "e2e_ms": e2e_ms, "job": j,
                   "perf": perf, "stages_ms": stages, "stages_from": "perf_log" if perf else "server_log",
                   "server_log_stages_ms": log_stages,
                   "pipeline_total_ms": (perf or {}).get("total_duration_ms") or (sum(log_stages.values()) if log_stages else None)}
            raw["requests"].append(rec)
            log(f"{role} {i}: e2e {e2e_ms:.1f} ms; stages {json.dumps({k: round(v, 1) for k, v in rec['stages_ms'].items()})}")
            (out / "results" / "raw.json").write_text(json.dumps(raw, indent=1))
    finally:
        log("stopping server")
        if profile:
            r = subprocess.run(["nsys", "shutdown", f"--session={session}"], env=env, capture_output=True, text=True)
            log(f"nsys shutdown rc={r.returncode} {r.stdout.strip()[:200]!r} {r.stderr.strip()[:200]!r}")
            for _ in range(60):
                if list((out / "profile").glob("*.nsys-rep")):
                    break
                time.sleep(2)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            for _ in range(60):
                if proc.poll() is not None:
                    break
                time.sleep(1)
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        raw["server_returncode"] = proc.returncode
        for _ in range(30):
            if port_free(port):
                break
            time.sleep(1)
        raw["port_released"] = port_free(port)
        log(f"port {port} released: {raw['port_released']}")
        if profile:
            for rep in sorted((out / "profile").glob("*.nsys-rep")):
                subprocess.run(["nsys", "export", "--type=sqlite", f"--output={rep.with_suffix('.sqlite')}",
                                "--force-overwrite=true", str(rep)], env=env, check=False)
                log(f"exported {rep.name}")
        (out / "results" / "raw.json").write_text(json.dumps(raw, indent=1))
    timed = [r for r in raw["requests"] if r["role"] == "timed"]
    names = sorted({n for r in timed for n in r["stages_ms"]})
    summary = {
        "id": cfg["id"], "hostname": raw["hostname"], "slurm_job_id": raw["slurm_job_id"],
        "quality": cfg.get("quality"), "numa": numa, "startup_s": raw.get("startup_s"),
        "e2e_ms": stats([r["e2e_ms"] for r in timed]),
        "pipeline_total_ms": stats([r["pipeline_total_ms"] for r in timed if r["pipeline_total_ms"] is not None]) if timed and timed[0]["pipeline_total_ms"] is not None else None,
        "stages_ms": {n: stats([r["stages_ms"][n] for r in timed if n in r["stages_ms"]]) for n in names},
        "outside_pipeline_ms": stats([r["e2e_ms"] - r["pipeline_total_ms"] for r in timed if r["pipeline_total_ms"] is not None]) if timed and timed[0]["pipeline_total_ms"] is not None else None,
    }
    (out / "results" / "summary.json").write_text(json.dumps(summary, indent=1))
    log("summary: e2e median %.1f ms; stages %s" % (summary["e2e_ms"]["median"],
        json.dumps({n: round(s["median"], 1) for n, s in summary["stages_ms"].items()})))


if __name__ == "__main__":
    main()
