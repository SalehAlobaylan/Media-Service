"""Managed external-process execution for Media-owned helpers."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from collections.abc import Callable, Sequence


_OUTPUT_TAIL_BYTES = 64 * 1024


async def _drain_stream(
    stream: asyncio.StreamReader | None,
    on_progress: Callable[[bytes], bool] | None,
    mark_progress: Callable[[], None],
) -> bytes:
    if stream is None:
        return b""
    tail = bytearray()
    while line := await stream.readline():
        tail.extend(line)
        if len(tail) > _OUTPUT_TAIL_BYTES:
            del tail[:-_OUTPUT_TAIL_BYTES]
        if on_progress is not None and on_progress(line):
            mark_progress()
    return bytes(tail)


async def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:  # pragma: no cover - Windows is not a supported production target.
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
        return
    except asyncio.TimeoutError:
        pass
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:  # pragma: no cover
        process.kill()
    await asyncio.wait_for(process.wait(), timeout=5)


async def run_managed_process(
    command: Sequence[str],
    *,
    timeout_sec: float,
    cwd: str | None = None,
    no_progress_timeout_sec: float | None = None,
    progress_parser: Callable[[bytes], bool] | None = None,
) -> tuple[int, bytes, bytes]:
    """Run a child with bounded timeout and complete process-group cleanup."""
    if not command:
        raise ValueError("managed process command cannot be empty")
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        start_new_session=(os.name == "posix"),
    )
    started = time.monotonic()
    last_progress = started

    def mark_progress() -> None:
        nonlocal last_progress
        last_progress = time.monotonic()

    stdout_task = asyncio.create_task(
        _drain_stream(process.stdout, progress_parser, mark_progress)
    )
    stderr_task = asyncio.create_task(
        _drain_stream(process.stderr, progress_parser, mark_progress)
    )
    wait_task = asyncio.create_task(process.wait())
    try:
        while not wait_task.done():
            now = time.monotonic()
            if now - started >= timeout_sec:
                raise TimeoutError(f"{command[0]} exceeded {timeout_sec}s")
            if (
                no_progress_timeout_sec is not None
                and now - last_progress >= no_progress_timeout_sec
            ):
                raise TimeoutError(
                    f"{command[0]} produced no meaningful progress for "
                    f"{no_progress_timeout_sec}s"
                )
            await asyncio.wait(
                {wait_task},
                timeout=min(
                    0.5,
                    max(0.01, timeout_sec - (now - started)),
                    max(
                        0.01,
                        no_progress_timeout_sec - (now - last_progress),
                    )
                    if no_progress_timeout_sec is not None
                    else 0.5,
                ),
            )
        returncode = await wait_task
        stdout, stderr = await asyncio.gather(stdout_task, stderr_task)
    except asyncio.CancelledError:
        await _terminate_process_group(process)
        raise
    except TimeoutError:
        await _terminate_process_group(process)
        raise
    finally:
        if process.returncode is None:
            await _terminate_process_group(process)
        if not wait_task.done():
            wait_task.cancel()
        for task in (stdout_task, stderr_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(wait_task, stdout_task, stderr_task, return_exceptions=True)
    return returncode or 0, stdout, stderr
