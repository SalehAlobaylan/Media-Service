"""Bounded extraction of one audio window from an approved long-form URL."""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

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
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
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
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_sec)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
        raise TimeoutError(f"audio segment extraction exceeded {timeout_sec}s")
    if process.returncode != 0:
        detail = (stderr or stdout).decode("utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"ffmpeg audio segment extraction failed: {detail}")
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError("ffmpeg produced an empty audio segment")
    return str(output)
