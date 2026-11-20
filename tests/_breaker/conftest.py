# Copyright 2021-2025 Zenauth Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cerbos.sdk._breaker import CircuitBreaker


class FakeClock:
    """A clock the tests move by hand, so the recovery window costs no sleeping."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _breaker_factory(cls: type[CircuitBreaker]):
    """Build breakers of `cls` on the test clock, with the Hub store's tuning scaled down."""

    def build(clock: FakeClock, **kwargs) -> CircuitBreaker:
        kwargs.setdefault("failure_threshold", 3)
        kwargs.setdefault("recovery_timeout", 60)
        kwargs.setdefault("expected_exception", ValueError)
        return cls(clock=clock, **kwargs)

    return build


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def make_breaker():
    return _breaker_factory(CircuitBreaker)


@pytest.fixture
def make_async_breaker():
    # Imported here so the sync tests keep collecting while the async class is missing.
    from cerbos.sdk._breaker import AsyncCircuitBreaker

    return _breaker_factory(AsyncCircuitBreaker)
