"""
Merge results from parallel batch scans into a single Telegram notification.

Called by the GitHub Actions merge job after all scan batches complete:
    autopilot merge --artifacts-dir artifacts/ --total-batches 3 --min-score 80 --top-n 5

Also usable locally after running batches manually:
    autopilot merge --artifacts-dir . --total-batches 3
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from job_hunt.log import get_logger
from job_hunt.notifier import send_telegram
from job_hunt.scanner import format_telegram_message

logger = get_logger()


def load_batch_jobs(artifacts_dir: Path, total_batches: int) -> list[dict]:
    """
    Load and deduplicate jobs from every batch state directory.

    actions/download-artifact@v4 nests each artifact under its own name:
        artifacts_dir/batch-N/state_batch_N/last_scan.json

    Falls back to a flat layout for local testing:
        artifacts_dir/state_batch_N/last_scan.json
    """
    all_jobs: list[dict] = []
    seen_urls: set[str] = set()

    for batch_idx in range(total_batches):
        candidates = [
            artifacts_dir / f"batch-{batch_idx}" / f"state_batch_{batch_idx}" / "last_scan.json",
            artifacts_dir / f"state_batch_{batch_idx}" / "last_scan.json",
        ]
        scan_file = next((p for p in candidates if p.exists()), None)
        if scan_file is None:
            logger.warning(f"Batch {batch_idx}: no last_scan.json found — artifact may have failed")
            continue

        try:
            batch_jobs: list[dict] = json.loads(scan_file.read_text(encoding="utf-8"))
            added = 0
            for job in batch_jobs:
                url = job.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_jobs.append(job)
                    added += 1
            logger.info(f"Batch {batch_idx}: {added} unique jobs loaded from {scan_file}")
        except Exception as exc:
            logger.error(f"Batch {batch_idx}: failed to load {scan_file}: {exc}")

    return all_jobs


def merge_and_notify(
    artifacts_dir: Path,
    total_batches: int,
    min_score: int,
    top_n: int,
    tg_token: str,
    tg_chat_id: str,
) -> None:
    """Merge all batch results and send exactly one Telegram notification."""
    logger.info(f"Merging {total_batches} batch(es) from {artifacts_dir}")

    all_jobs = load_batch_jobs(artifacts_dir, total_batches)
    logger.info(f"Total unique jobs collected: {len(all_jobs)}")

    top_jobs = sorted(
        [j for j in all_jobs if j.get("score", 0) >= min_score],
        key=lambda x: x.get("score", 0),
        reverse=True,
    )[:top_n]

    date_str = datetime.now().strftime("%d %b %Y")
    telegram_configured = bool(tg_token and tg_chat_id)

    if not top_jobs:
        logger.info("No matching jobs found across all batches.")
        if telegram_configured:
            send_telegram(tg_token, tg_chat_id, f"<b>Job Hunt — {date_str}</b>\nNo new matches today.")
        return

    logger.info(f"Top {len(top_jobs)} jobs (score >= {min_score}):")
    for j in top_jobs:
        logger.info(
            f"  [{j.get('score', '?'):3}] "
            f"{j.get('extracted_title') or j.get('title')} @ {j.get('company')}"
        )

    msg = format_telegram_message(top_jobs, date_str)
    logger.info(f"\n{msg}")

    if telegram_configured:
        sent = send_telegram(tg_token, tg_chat_id, msg)
        logger.info("Telegram notification sent." if sent else "Telegram send failed.")
    else:
        logger.info("Telegram not configured — skipping notification.")


def _parse_args(argv: list[str]) -> dict:
    """Parse CLI arguments for the merge command."""
    args: dict = {
        "artifacts_dir": Path("artifacts"),
        "total_batches": 3,
        "min_score": int(os.getenv("MIN_SCORE", "60")),
        "top_n": int(os.getenv("TOP_N", "5")),
        "tg_token": os.getenv("TELEGRAM_TOKEN", ""),
        "tg_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
    }
    flag_map = {
        "--artifacts-dir": ("artifacts_dir", Path),
        "--total-batches": ("total_batches", int),
        "--min-score": ("min_score", int),
        "--top-n": ("top_n", int),
    }
    for flag, (key, cast) in flag_map.items():
        if flag in argv:
            idx = argv.index(flag)
            try:
                args[key] = cast(argv[idx + 1])
            except (IndexError, ValueError):
                sys.exit(f"{flag} requires a value (e.g. {flag} 3)")
    return args


def run_merge(argv: list[str]) -> None:
    """Entry point called by `autopilot merge <args>`."""
    args = _parse_args(argv)
    merge_and_notify(**args)


if __name__ == "__main__":
    run_merge(sys.argv[1:])
