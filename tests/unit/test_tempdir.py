from pathlib import Path

import pytest

from src.utils.tempdir import resolve_media_temp_dir


def test_tempdir_is_created_and_left_clean(tmp_path: Path) -> None:
    resolved = Path(resolve_media_temp_dir(str(tmp_path / "media-scratch")))

    assert resolved.is_dir()
    assert list(resolved.iterdir()) == []


def test_tempdir_rejects_a_file(tmp_path: Path) -> None:
    path = tmp_path / "not-a-directory"
    path.write_text("nope")

    with pytest.raises(FileExistsError):
        resolve_media_temp_dir(str(path))
