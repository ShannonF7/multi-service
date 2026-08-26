"""Standalone semantic completion worker.

Run under systemd, for example:
  python -m src.rag.semantic_completion_worker --worker-id semantic-worker-1
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import time
import uuid

from src.rag.service.completion_job_service import (
    claim_next_semantic_completion_job,
    recover_stale_semantic_completion_jobs,
    run_semantic_completion_job,
)

from src.rag.service.graph_sync_service import (
    claim_next_graph_sync_job,
    process_graph_sync_job,
    recover_stale_graph_sync_jobs,
)
from src.rag.service.graph_growth_service import (
    claim_next_graph_discovery_job,
    enqueue_graph_discovery_for_sync,
    process_graph_discovery_job,
    recover_stale_graph_discovery_jobs,
    refresh_graph_discovery_validation_statuses,
)
logger = logging.getLogger(__name__)
_STOP = False


def _handle_stop(signum, frame):  # type: ignore[no-untyped-def]
    global _STOP
    _STOP = True
    logger.info("semantic completion worker received stop signal: %s", signum)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic completion job worker")
    parser.add_argument("--worker-id", default=os.getenv("SEMANTIC_WORKER_ID") or f"semantic-worker-{uuid.uuid4().hex[:8]}")
    parser.add_argument("--poll-interval", type=float, default=float(os.getenv("SEMANTIC_WORKER_POLL_INTERVAL", "2")))
    parser.add_argument("--lease-seconds", type=int, default=int(os.getenv("SEMANTIC_WORKER_LEASE_SECONDS", "900")))
    parser.add_argument("--recover-every", type=int, default=int(os.getenv("SEMANTIC_WORKER_RECOVER_EVERY", "30")))
    parser.add_argument("--log-level", default=os.getenv("SEMANTIC_WORKER_LOG_LEVEL", "INFO"))
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    logger.info("semantic completion worker started worker_id=%s", args.worker_id)
    last_recover = 0.0
    while not _STOP:
        now = time.time()
        if now - last_recover >= max(5, args.recover_every):
            recovered = recover_stale_semantic_completion_jobs(worker_id=args.worker_id)
            if recovered:
                logger.warning("recovered stale semantic completion jobs: %s", recovered)
            graph_recovered = recover_stale_graph_sync_jobs()
            if graph_recovered:
                logger.warning("recovered stale graph sync jobs: %s", graph_recovered)
            discovery_recovered = recover_stale_graph_discovery_jobs()
            if discovery_recovered:
                logger.warning("recovered stale graph discovery jobs: %s", discovery_recovered)
            refreshed = refresh_graph_discovery_validation_statuses()
            if refreshed:
                logger.info("refreshed graph discovery validation statuses: %s", refreshed)
            last_recover = now
        graph_job = claim_next_graph_sync_job(worker_id=args.worker_id, lease_seconds=min(args.lease_seconds, 300))
        if graph_job:
            graph_job_id = int(graph_job["id"])
            logger.info("claimed graph sync job id=%s event=%s", graph_job_id, graph_job.get("event_key"))
            try:
                process_graph_sync_job(graph_job_id, worker_id=args.worker_id)
            except Exception:
                logger.exception("graph sync job failed id=%s", graph_job_id)
            else:
                try:
                    discovery_job = enqueue_graph_discovery_for_sync(graph_job)
                    if discovery_job:
                        logger.info(
                            "enqueued graph discovery job id=%s from graph sync id=%s reused=%s",
                            discovery_job.get("id"),
                            graph_job_id,
                            discovery_job.get("reused"),
                        )
                except Exception:
                    logger.exception("enqueue graph discovery failed graph_sync_job_id=%s", graph_job_id)
        discovery_job = claim_next_graph_discovery_job(
            worker_id=args.worker_id,
            lease_seconds=min(args.lease_seconds, 600),
        )
        if discovery_job:
            discovery_job_id = int(discovery_job["id"])
            logger.info(
                "claimed graph discovery job id=%s event=%s",
                discovery_job_id,
                discovery_job.get("event_key"),
            )
            try:
                process_graph_discovery_job(discovery_job_id, worker_id=args.worker_id)
            except Exception:
                logger.exception("graph discovery job failed id=%s", discovery_job_id)
        job = claim_next_semantic_completion_job(worker_id=args.worker_id, lease_seconds=args.lease_seconds)
        if not job:
            if not graph_job and not discovery_job:
                await asyncio.sleep(max(0.5, args.poll_interval))
            continue
        job_id = int(job["id"])
        logger.info("claimed semantic completion job id=%s trace=%s", job_id, job.get("trace_id"))
        await run_semantic_completion_job(job_id, worker_id=args.worker_id, already_claimed=True)
    logger.info("semantic completion worker stopped worker_id=%s", args.worker_id)


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
