"""Per-GPU bf16 GEMM throughput + clocks: catches a thermally throttled or otherwise slow GPU."""
import subprocess, time, torch
for d in range(torch.cuda.device_count()):
    torch.cuda.set_device(d)
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    for _ in range(5): a @ b
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(50): a @ b
    torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 50
    q = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw,clocks_event_reasons.active", "--format=csv,noheader", "-i", str(d)], capture_output=True, text=True).stdout.strip()
    print(f"gpu{d}: {2*8192**3/dt/1e12:7.1f} TFLOPS bf16 8192^3 | {q}", flush=True)
    del a, b; torch.cuda.empty_cache()
