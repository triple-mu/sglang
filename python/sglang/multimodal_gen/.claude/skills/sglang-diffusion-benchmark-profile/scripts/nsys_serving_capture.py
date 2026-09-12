#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Serve, warm up with real requests, then capture one steady-state request in Nsight Systems.

`sglang serve` runs under `nsys launch` (profiler injected, collection idle).
Once /health reports ready the script sends the warmup requests through
/v1/videos, then brackets the capture request(s) with `nsys start` and
`nsys stop`, so the report holds exactly the steady-state end-to-end request
for every rank without any per-process capture range. Run the server with
`--enable-nvtx-marker true` so the report carries `request#<n>` and stage
ranges (`--enable-layerwise-nvtx-marker` adds per-module ranges at a cost).

The request comes from a `sglang generate` config: its ServerArgs fields
(model_variant, num_gpus, ...) become `sglang serve` flags unless the same
flag is passed after `--`, the remaining fields form the request body, and
relative media paths in `conditions[*].uri` are resolved to file URIs next to
the config. One config therefore serves t2va, fl2va and ref2va alike:

  python nsys_serving_capture.py --request-json /workspace/benchmark_inputs/ref2va/input.json \\
      --output /workspace/nsys/h3_ref2va --warmup-requests 1 \\
      -- --model-path MiniMaxAI/MiniMax-H3 --num-gpus 4 --tp-size 1 \\
         --ulysses-degree 4 --warmup-mode off --enable-nvtx-marker true
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shlex
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.argparse import (
    FlexibleArgumentParser,
    StoreBoolean,
)

TERMINAL_STATUSES = frozenset({"completed", "failed", "deleted"})
# generate-only delivery knobs; the server names and stores the output itself
GENERATE_ONLY_FIELDS = frozenset(
    {"save_output", "return_file_paths_only", "output_file_name"}
)
# The script owns these; a generate config must not steer the server it profiles.
SERVE_FIELDS_NOT_FROM_CONFIG = frozenset({"host", "port", "output_path"})
# Condition URIs the server fetches or decodes itself.
REMOTE_URI_SCHEMES = ("http", "https", "data", "base64", "tar+offset", "tar+b64header")
# sglang serve falls back to a free port when the requested one is taken and
# announces the one it bound; the log line carries ANSI colour codes.
_UVICORN_RE = re.compile(r"Uvicorn running on .*?http://[\d.]+:(\d+)")


def _log(message: str) -> None:
    print(f"[nsys-serving] {message}", flush=True)


def resolve_conditions(config: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Point relative media paths in `conditions[*].uri` at absolute file URIs,
    as the `sglang generate` wrappers do before submitting the config."""
    if "conditions" not in config:
        return dict(config)
    conditions = []
    for item in config.get("conditions") or []:
        item = dict(item)
        uri = str(item.get("uri", ""))
        if uri.startswith(tuple(f"{scheme}:" for scheme in REMOTE_URI_SCHEMES)):
            conditions.append(item)
            continue
        if uri.startswith("file:"):
            candidate = Path(urllib.parse.unquote(urllib.parse.urlparse(uri).path))
            if not candidate.is_file():
                candidate = base_dir / candidate.name
        else:
            candidate = base_dir / uri
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"condition media not found: {candidate}")
        item["uri"] = candidate.as_uri()
        conditions.append(item)
    return {**config, "conditions": conditions}


def request_payload(
    config: dict[str, Any], *, output_dir: Path, perf_dump: Path
) -> dict:
    """The /v1/videos body for a `sglang generate` config: every key that is not
    a ServerArgs field stays a request field (prompt, task, conditions, ...)."""
    server_fields = {field.name for field in dataclasses.fields(ServerArgs)}
    payload = {
        key: value
        for key, value in config.items()
        if key not in server_fields and key not in GENERATE_ONLY_FIELDS
    }
    payload["output_path"] = str(output_dir)
    payload["perf_dump_path"] = str(perf_dump)
    return payload


def serve_args_from_config(config: dict[str, Any], explicit: list[str]) -> list[str]:
    """`sglang serve` flags for the ServerArgs fields of a generate config.

    Flags already present in `explicit` (the arguments after `--`) win, so a
    config written for eight GPUs can be served on four by passing
    `--num-gpus 4`.
    """
    parser = FlexibleArgumentParser()
    ServerArgs.add_cli_args(parser)
    actions = {
        action.dest: action for action in parser._actions if action.option_strings
    }
    explicit_flags = {
        arg.split("=", 1)[0].replace("_", "-")
        for arg in explicit
        if arg.startswith("--")
    }
    derived: list[str] = []
    for key, value in config.items():
        action = actions.get(key)
        if action is None or value is None or key in SERVE_FIELDS_NOT_FROM_CONFIG:
            continue
        flag = action.option_strings[-1]
        if flag in explicit_flags:
            continue
        if isinstance(action, StoreBoolean):
            derived += [flag, "true" if value else "false"]
        elif isinstance(action, argparse._StoreConstAction):
            if value == action.const:
                derived.append(flag)
        elif isinstance(value, list):
            derived += [flag, *(str(item) for item in value)]
        else:
            derived += [flag, str(value)]
    return derived


def _http(method: str, url: str, body: dict | None = None, timeout: float = 60.0):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
        return response.status, (json.loads(raw) if raw else {})


class Server:
    def __init__(self, args: argparse.Namespace, log_path: Path):
        self.args = args
        self.port = args.port
        self.base_url = f"http://{args.host}:{args.port}"
        self.log_path = log_path
        self.session = args.session or f"sglang-{os.getpid()}"
        self.process: subprocess.Popen | None = None
        self._port_confirmed = False

    def nsys(self, *parts: str) -> list[str]:
        return [self.args.nsys, *parts]

    def launch(self, serve_args: list[str]) -> None:
        command = [
            *self.nsys(
                "launch",
                f"--session={self.session}",
                f"--trace={self.args.trace}",
                "--trace-fork-before-exec=true",
            ),
            *shlex.split(self.args.nsys_launch_args),
            "sglang",
            "serve",
            "--host",
            self.args.host,
            "--port",
            str(self.args.port),
            *serve_args,
        ]
        env = dict(os.environ)
        # nsys records only registered NVTX strings by default; the markers are plain.
        env.setdefault("NSYS_NVTX_PROFILER_REGISTER_ONLY", "0")
        for item in self.args.env:
            key, _, value = item.partition("=")
            env[key] = value
        _log("launching: " + " ".join(shlex.quote(part) for part in command))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # A fresh log per launch: the bound port is read back from it.
        log = open(self.log_path, "wb")
        # Own session so the nsys launcher, the HTTP server and every worker
        # process can be stopped together.
        self.process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

    def _sync_port_from_log(self) -> None:
        if self._port_confirmed or not self.log_path.exists():
            return
        matches = _UVICORN_RE.findall(self.log_path.read_text(errors="replace"))
        if not matches:
            return
        port = int(matches[-1])
        if port != self.port:
            _log(f"server bound port {port}; requested port {self.port} was taken")
            self.port = port
            self.base_url = f"http://{self.args.host}:{port}"
        self._port_confirmed = True

    def wait_ready(self) -> None:
        started = time.monotonic()
        deadline = started + self.args.health_timeout
        while time.monotonic() < deadline:
            self._sync_port_from_log()
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"server exited with {self.process.returncode} before /health; "
                    f"see {self.log_path}"
                )
            try:
                status, _ = _http("GET", f"{self.base_url}/health", timeout=10)
                if status == 200:
                    _log(f"server ready after {time.monotonic() - started:.0f} s")
                    return
            except (urllib.error.URLError, urllib.error.HTTPError, OSError):
                pass
            time.sleep(5)
        raise TimeoutError(f"/health not ready after {self.args.health_timeout} s")

    def run_request(self, payload: dict, label: str) -> dict:
        started = time.monotonic()
        status, job = _http("POST", f"{self.base_url}/v1/videos", payload, timeout=120)
        job_id = job["id"]
        _log(f"{label}: submitted {job_id} (HTTP {status})")
        deadline = time.monotonic() + self.args.request_timeout
        while time.monotonic() < deadline:
            time.sleep(2)
            _, job = _http("GET", f"{self.base_url}/v1/videos/{job_id}", timeout=30)
            if job.get("status") in TERMINAL_STATUSES:
                break
        else:
            raise TimeoutError(f"{label}: {job_id} still {job.get('status')}")
        elapsed = time.monotonic() - started
        _log(f"{label}: {job_id} {job.get('status')} in {elapsed:.1f} s")
        if job.get("status") != "completed":
            raise RuntimeError(f"{label}: request failed: {job.get('error')}")
        job["wall_seconds"] = elapsed
        return job

    def capture_start(self, report: Path) -> None:
        for stale in (report, report.with_suffix(".sqlite")):
            if stale.exists():
                stale.unlink()
        command = [
            *self.nsys(
                "start",
                f"--session={self.session}",
                f"--output={report}",
                "--force-overwrite=true",
                "--sample=none",
                "--cpuctxsw=none",
            ),
            *shlex.split(self.args.nsys_start_args),
        ]
        _log("nsys start: " + " ".join(shlex.quote(part) for part in command))
        subprocess.run(command, check=True)

    def capture_stop(self) -> None:
        _log("nsys stop")
        subprocess.run(self.nsys("stop", f"--session={self.session}"), check=True)

    def shutdown(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        _log("stopping server")
        os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=30)


def perf_summary(perf_dump: Path) -> str:
    if not perf_dump.exists():
        return "no perf dump"
    dump = json.loads(perf_dump.read_text())
    stages = ", ".join(
        f"{step['name']}={step['duration_ms']:.0f}"
        for step in dump.get("steps", [])
        if step.get("duration_ms", 0) >= 50
    )
    return f"total {dump.get('total_duration_ms', 0):.0f} ms; {stages}"


def report_summary(nsys: str, report: Path) -> None:
    """Per-device kernel coverage and the request/stage NVTX ranges of the report,
    so a rank that dropped out of the capture is visible right away."""
    sqlite_path = report.with_suffix(".sqlite")
    subprocess.run(
        [
            nsys,
            "export",
            "--type=sqlite",
            "--force-overwrite=true",
            f"--output={sqlite_path}",
            str(report),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    db = sqlite3.connect(sqlite_path)
    try:
        for device, count, busy in db.execute(
            "SELECT deviceId, COUNT(*), SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL "
            "GROUP BY deviceId ORDER BY deviceId"
        ):
            _log(f"device {device}: {count} kernels, {busy / 1e9:.1f} s busy")
        for text, count, average in db.execute(
            "SELECT text, COUNT(*), AVG(end - start) FROM NVTX_EVENTS "
            "WHERE text LIKE 'request#%' OR text LIKE 'stage_%' "
            "GROUP BY text ORDER BY MIN(start)"
        ):
            _log(f"nvtx {text}: {count} ranges, {average / 1e9:.2f} s each")
    except sqlite3.OperationalError as exc:
        _log(f"report summary unavailable: {exc}")
    finally:
        db.close()


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--request-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--capture-requests", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--nsys", default="nsys")
    parser.add_argument("--session", default=None)
    parser.add_argument("--trace", default="cuda,nvtx,cublas,cudnn")
    parser.add_argument("--nsys-launch-args", default="")
    parser.add_argument("--nsys-start-args", default="")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--health-timeout", type=float, default=1800.0)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument("--no-report-summary", action="store_true")
    argv = sys.argv[1:]
    serve_args: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, serve_args = argv[:split], argv[split + 1 :]
    return parser.parse_args(argv), serve_args


def main() -> int:
    args, explicit_serve_args = parse_args()
    config = resolve_conditions(
        json.loads(args.request_json.read_text()), args.request_json.resolve().parent
    )
    serve_args = [
        *serve_args_from_config(config, explicit_serve_args),
        *explicit_serve_args,
    ]
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / "steady_state.nsys-rep"
    server = Server(args, output_dir / "server.log")
    server.launch(serve_args)
    results = []
    try:
        server.wait_ready()
        for index in range(args.warmup_requests):
            perf_dump = output_dir / f"perf_warmup{index + 1}.json"
            payload = request_payload(
                config, output_dir=output_dir, perf_dump=perf_dump
            )
            job = server.run_request(payload, f"warmup request {index + 1}")
            results.append((f"warmup {index + 1}", job["wall_seconds"], perf_dump))
        server.capture_start(report)
        try:
            for index in range(args.capture_requests):
                perf_dump = output_dir / f"perf_capture{index + 1}.json"
                payload = request_payload(
                    config, output_dir=output_dir, perf_dump=perf_dump
                )
                job = server.run_request(payload, f"captured request {index + 1}")
                results.append(
                    (f"captured {index + 1}", job["wall_seconds"], perf_dump)
                )
        finally:
            server.capture_stop()
    finally:
        if not args.keep_server:
            server.shutdown()
    _log(f"report: {report}")
    for label, wall, perf_dump in results:
        _log(f"{label}: wall {wall:.1f} s; {perf_summary(perf_dump)}")
    if not args.no_report_summary:
        report_summary(args.nsys, report)
    _log(
        "kernel summary: "
        f"{args.nsys} stats --report cuda_gpu_kern_sum --format csv {report}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
