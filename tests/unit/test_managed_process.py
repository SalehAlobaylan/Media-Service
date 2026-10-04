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


@pytest.mark.asyncio
async def test_managed_launch_registers_helper_and_cleans_after_timeout(tmp_path, monkeypatch) -> None:
    import asyncio
    import json
    monkeypatch.setenv("WAHB_MANAGED_ENV", "1")
    monkeypatch.setenv("WAHB_PROCESS_REGISTRY_DIR", str(tmp_path))
    monkeypatch.setenv("WAHB_RUNTIME_ID", "fixture")
    monkeypatch.setenv("WAHB_PROCESS_OWNER_TOKEN", "fixture-owner")
    task = asyncio.create_task(run_managed_process([sys.executable, "-c", "import time; time.sleep(5)"], timeout_sec=1))
    for _ in range(50):
        markers = list(tmp_path.glob("*.json"))
        if markers:
            break
        await asyncio.sleep(.01)
    assert markers
    record = json.loads(markers[0].read_text())
    assert record["runtime_id"] == "fixture"
    assert record["process_start"]
    assert markers[0].stat().st_mode & 0o077 == 0
    with pytest.raises(TimeoutError):
        await task
    assert list(tmp_path.glob("*.json")) == []
