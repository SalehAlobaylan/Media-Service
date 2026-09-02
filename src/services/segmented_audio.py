"""Bounded extraction of one audio window from an approved long-form URL."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from src.common.managed_process import run_managed_process
from src.utils.url_guard import UnsafeURLError, validate_public_url


async def extract_audio_segment(
    url: str,
    output_path: str,
    start_ms: int,
    end_ms: int,
    *,
    timeout_sec: int = 1800,
    trusted_cms_source: bool = False,
) -> str:
    if end_ms <= start_ms or start_ms < 0:
        raise ValueError("invalid audio segment bounds")
    if not trusted_cms_source:
        try:
            validate_public_url(url)
        except UnsafeURLError as exc:
            raise ValueError(f"refusing to fetch unsafe long-form URL: {exc}") from exc

    duration_sec = (end_ms - start_ms) / 1000
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    latest_progress_us = -1

    def ffmpeg_progress(line: bytes) -> bool:
        nonlocal latest_progress_us
        match = re.fullmatch(rb"out_time_(?:us|ms)=(\d+)\s*", line)
        if match is None:
            return False
        value = int(match.group(1))
        if value <= latest_progress_us:
            return False
        latest_progress_us = value
        return True

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-threads",
        "2",
        "-filter_threads",
        "1",
        "-progress",
        "pipe:2",
        "-nostats",
        "-ss",
        f"{start_ms / 1000:.3f}",
        "-i",
        url,
        "-t",
        f"{duration_sec:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-y",
        str(output),
    ]
    try:
        returncode, stdout, stderr = await run_managed_process(
            command,
            timeout_sec=timeout_sec,
            no_progress_timeout_sec=min(180, max(30, duration_sec)),
            progress_parser=ffmpeg_progress,
        )
    except asyncio.CancelledError:
        # Cancellation is a worker lifecycle signal, not a media timeout. The
        # supervisor has already terminated the process group; preserve the
        # cancellation so the caller can release its lease and retry safely.
        raise
    except TimeoutError:
        raise TimeoutError(f"audio segment extraction exceeded {timeout_sec}s")
    if returncode != 0:
        detail = (stderr or stdout).decode("utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"ffmpeg audio segment extraction failed: {detail}")
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError("ffmpeg produced an empty audio segment")
    return str(output)
