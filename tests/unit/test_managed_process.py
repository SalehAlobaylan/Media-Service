import sys

import pytest

from src.common.managed_process import run_managed_process


@pytest.mark.asyncio
async def test_managed_process_preserves_bounded_output_tail() -> None:
    code, stdout, stderr = await run_managed_process(
        [sys.executable, "-c", "print('ready')"], timeout_sec=2
    )
    assert code == 0
    assert stdout == b"ready\n"
    assert stderr == b""


@pytest.mark.asyncio
async def test_managed_process_rejects_noise_without_meaningful_progress() -> None:
    with pytest.raises(TimeoutError, match="no meaningful progress"):
        await run_managed_process(
            [
                sys.executable,
                "-c",
                "import sys,time\nwhile True:\n print('warning', file=sys.stderr, flush=True); time.sleep(.01)",
            ],
            timeout_sec=2,
            no_progress_timeout_sec=0.05,
            progress_parser=lambda line: line.startswith(b"progress="),
        )
