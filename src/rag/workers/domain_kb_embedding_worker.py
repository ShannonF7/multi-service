"""Domain KB embedding worker.

Run from project root:
  conda activate llama_factory
  python -m src.rag.workers.domain_kb_embedding_worker --worker-id kb-embed-1

Optional fixed GPU:
  EMBEDDING_DEVICE=cuda:0 python -m src.rag.workers.domain_kb_embedding_worker
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import time

from src.rag.service.embedding_job_service import process_one_embedding_job


def choose_gpu_device() -> str:
    fixed = os.getenv("EMBEDDING_DEVICE", "").strip()
    if fixed:
        return fixed
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        candidates = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            idx, mem_used, mem_total, util = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            mem_ratio = mem_used / max(mem_total, 1)
            score = mem_ratio * 100 + util
            candidates.append((score, idx))
        if candidates:
            candidates.sort()
            return f"cuda:{candidates[0][1]}"
    except Exception:
        pass
    return "cuda:0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", default=f"kb-embed-{socket.gethostname()}-{os.getpid()}")
    parser.add_argument("--sleep", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        device = choose_gpu_device()
        os.environ["EMBEDDING_DEVICE"] = device
        result = process_one_embedding_job(worker_id=args.worker_id, device=device)
        if result:
            print(result, flush=True)
        elif args.once:
            print({"status": "idle", "device": device}, flush=True)
            return
        if args.once:
            return
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
