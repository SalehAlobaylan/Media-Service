import asyncio
import time
from collections.abc import Callable, Coroutine
from enum import Enum
from typing import Any, TypeVar

from src.middleware.error_handler import CircuitOpenError
from src.utils.logging import get_logger
from src.utils.metrics import circuit_state

logger = get_logger(__name__)
T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = 0
    OPEN = 1
    HALF_OPEN = 2


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout_sec: int = 30,
        half_open_requests: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout_sec = reset_timeout_sec
        self.half_open_requests = half_open_requests

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def execute(
        self, func: Callable[..., Coroutine[Any, Any, T]], *args: Any, **kwargs: Any
    ) -> T:
        async with self._lock:
            self._check_state_transition()

            if self._state == CircuitState.OPEN:
                raise CircuitOpenError(
                    f"Circuit breaker is OPEN. Retry after {self.reset_timeout_sec}s."
                )

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception:
            await self._on_failure()
            raise

    def _check_state_transition(self) -> None:
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.reset_timeout_sec:
                logger.info("circuit_half_open", elapsed_sec=round(elapsed, 1))
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                circuit_state.set(CircuitState.HALF_OPEN.value)

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_requests:
                    logger.info("circuit_closed", after_successes=self._success_count)
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    circuit_state.set(CircuitState.CLOSED.value)
            else:
                self._failure_count = 0

    async def _on_failure(self) -> None:
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                logger.warning("circuit_opened", reason="failure in half-open state")
                self._state = CircuitState.OPEN
                circuit_state.set(CircuitState.OPEN.value)
            elif self._failure_count >= self.failure_threshold:
                logger.warning("circuit_opened", failures=self._failure_count)
                self._state = CircuitState.OPEN
                circuit_state.set(CircuitState.OPEN.value)
