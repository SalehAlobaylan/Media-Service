from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import pytest
from PIL import Image
from tests.fakes import RecordingSTT
from src.services.image_embedding import ImageEmbeddingService
from src.services.transcription import TranscriptionService
from src.middleware.error_handler import TranscriptionError


@pytest.mark.asyncio
async def test_clip_records_executing_tenant_run_and_units_once():
    events = []
    clip = Mock(is_loaded=True, model_name="clip", dimensions=512)
    clip.encode_image.return_value = [0.1] * 512
    cms = SimpleNamespace(spend_meter=SimpleNamespace(enqueue=events.append))
    image = BytesIO()
    Image.new("RGB", (2, 2)).save(image, format="PNG")
    await ImageEmbeddingService(clip, cms).embed_bytes(
        image.getvalue(),
        spend_context={
            "tenant_id": "a",
            "trigger_source": "embedding_campaign",
            "system_run_id": "campaign-1",
            "economics_attribution_ref": "content:opaque-fixture",
        },
    )
    assert len(events) == 1 and events[0]["spend_class"] == "image_embed"
    assert events[0]["tenant_id"] == "a" and events[0]["system_run_id"] == "campaign-1"
    assert events[0]["economics_attribution_ref"] == "content:opaque-fixture"
    assert events[0]["units"]["items"] == 1 and events[0]["units"]["compute_ms"] >= 0


@pytest.mark.asyncio
async def test_local_whisper_is_compute_and_paid_stateless_has_no_provider_call():
    events = []
    stt = RecordingSTT()
    stt.name = "faster-whisper"
    cms = SimpleNamespace(spend_meter=SimpleNamespace(enqueue=events.append))
    await TranscriptionService(stt, cms).transcribe_file(
        "fixture.wav", segment_context={"tenant_id": "a", "system_run_id": "gen-1"}
    )
    assert (
        len(events) == 1
        and events[0]["spend_class"] == "stt_local"
        and events[0]["units"]["audio_sec"] == 1
    )
    paid = RecordingSTT()
    with pytest.raises(TranscriptionError):
        await TranscriptionService(paid, cms).transcribe_file("fixture.wav")
    assert not paid.calls


@pytest.mark.asyncio
async def test_paid_owner_decline_and_segment_exception_are_visible_without_double_billing():
    lost = []
    events = []
    stt = RecordingSTT()
    cms = SimpleNamespace(
        request_stt_owner=AsyncMock(
            return_value={"triggered": False, "reason": "budget_capped"}
        ),
        get_transcription_spend_context=AsyncMock(
            return_value={
                "tenant_id": "a",
                "status": "running",
                "system_run_id": "owner-job",
            }
        ),
        spend_meter=SimpleNamespace(
            enqueue=events.append, unobservable_attempt=lost.append
        ),
    )
    with pytest.raises(TranscriptionError):
        await TranscriptionService(stt, cms).transcribe_file(
            "fixture.wav", content_id="item"
        )
    assert not stt.calls
    await TranscriptionService(stt, cms).transcribe_file(
        "fixture.wav", segment_context={"transcription_job_id": "job"}
    )
    assert not events
    stt.transcribe = Mock(side_effect=RuntimeError("provider timeout"))
    with pytest.raises(RuntimeError):
        await TranscriptionService(stt, cms).transcribe_file(
            "fixture.wav", segment_context={"transcription_job_id": "job"}
        )
    assert lost == ["a"]


@pytest.mark.asyncio
async def test_terminal_meter_health_reports_closing_and_loss_separately():
    from src.services.spend_meter import SpendMeter

    health = []

    async def delivery(events, health=None):
        if health is not None:
            reported.append(health)
        return {"acknowledgements": [{"event_id": e["event_id"], "status": "inserted"} for e in events]}

    reported = health
    meter = SpendMeter(SimpleNamespace(emit_ai_spend_events=delivery))
    meter.enqueue({"event_id": "one", "tenant_id": "a"})
    await meter.publish_health()
    assert health[-1]["closed"] is False
    meter.start()
    await meter.close()
    assert health[-1]["closed"] is True
    assert health[-1]["accepted"] == 1 and health[-1]["dropped"] == 0
