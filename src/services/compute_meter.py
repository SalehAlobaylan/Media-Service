from datetime import UTC, datetime
from uuid import uuid4
from src.services.spend_meter import SpendMeter


def emit_compute(
    meter: SpendMeter | None, context: dict, spend_class: str, model: str, units: dict
) -> None:
    if meter is not None:
        meter.enqueue(
            {
                "event_id": str(uuid4()),
                "occurred_at": datetime.now(UTC).isoformat(),
                "spend_class": spend_class,
                "operation": spend_class,
                "provider": "local",
                "model": model,
                "units": units,
                "tenant_id": context.get("tenant_id") or "default",
                "trigger_source": context.get("trigger_source") or "unknown",
                "system_run_id": context.get("system_run_id") or "",
                "source_service": "media",
                "human_actor_id": context.get("human_actor_id") or "",
            }
        )
