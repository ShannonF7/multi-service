"""Dedicated worker for durable A-side to Neo4j graph projection jobs."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import time
import uuid

from src.rag.service.graph_growth_service import enqueue_graph_discovery_for_sync
from src.rag.service.graph_sync_service import (
    claim_next_graph_sync_job,
    process_graph_sync_job,
    recover_stale_graph_sync_jobs,
)


logger = logging.getLogger(__name__)
_STOP = False


def _handle_stop(signum, frame):  # type: ignore[no-untyped-def]
    global _STOP
    _STOP = True
    logger.info("graph sync worker received stop signal: %s", signum)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Published graph projection worker")
    parser.add_argument(
        "--worker-id",
        default=os.getenv("GRAPH_SYNC_WORKER_ID")
        or f"graph-sync-worker-{uuid.uuid4().hex[:8]}",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=float(os.getenv("GRAPH_SYNC_WORKER_POLL_INTERVAL", "1")),
    )
    parser.add_argument(
        "--lease-seconds",
        type=int,
        default=int(os.getenv("GRAPH_SYNC_WORKER_LEASE_SECONDS", "300")),
    )
    parser.add_argument(
        "--recover-every",
        type=int,
        default=int(os.getenv("GRAPH_SYNC_WORKER_RECOVER_EVERY", "30")),
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("GRAPH_SYNC_WORKER_LOG_LEVEL", "INFO"),
    )
    return parser.parse_args()


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )
    logger.info("graph sync worker started worker_id=%s", args.worker_id)
    last_recover = 0.0

    while not _STOP:
        now = time.time()
        if now - last_recover >= max(5, args.recover_every):
            recovered = recover_stale_graph_sync_jobs()
            if recovered:
                logger.warning("recovered stale graph sync jobs: %s", recovered)
            last_recover = now

        job = claim_next_graph_sync_job(
            worker_id=args.worker_id,
            lease_seconds=min(max(30, args.lease_seconds), 600),
        )
        if not job:
            time.sleep(max(0.2, args.poll_interval))
            continue

        job_id = int(job["id"])
        logger.info(
            "claimed graph sync job id=%s event=%s",
            job_id,
            job.get("event_key"),
        )
        try:
            process_graph_sync_job(job_id, worker_id=args.worker_id)
        except Exception:
            logger.exception("graph sync job failed id=%s", job_id)
            continue

        try:
            discovery_job = enqueue_graph_discovery_for_sync(job)
            if discovery_job:
                logger.info(
                    "enqueued graph discovery job id=%s from graph sync id=%s reused=%s",
                    discovery_job.get("id"),
                    job_id,
                    discovery_job.get("reused"),
                )
        except Exception:
            logger.exception(
                "enqueue graph discovery failed graph_sync_job_id=%s",
                job_id,
            )

    logger.info("graph sync worker stopped worker_id=%s", args.worker_id)


if __name__ == "__main__":
    main()
