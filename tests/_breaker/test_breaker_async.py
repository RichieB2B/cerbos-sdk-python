# Copyright 2021-2025 Zenauth Ltd.
# SPDX-License-Identifier: Apache-2.0

import inspect

import pytest

from cerbos.sdk._breaker import CircuitBreakerError

pytestmark = pytest.mark.anyio


async def open_the_circuit(breaker, wrapped) -> None:
    for _ in range(3):
        with pytest.raises(ValueError):
            await wrapped()

    assert breaker.opened


async def test_async_opens_at_threshold_not_before(clock, make_async_breaker):
    breaker = make_async_breaker(clock)

    @breaker.decorate
    async def fail():
        raise ValueError("boom")

    for _ in range(2):
        with pytest.raises(ValueError):
            await fail()

    assert breaker.closed
    assert breaker.failure_count == 2

    with pytest.raises(ValueError):
        await fail()

    assert breaker.opened
    assert breaker.failure_count == 3


async def test_async_open_circuit_does_not_call_the_wrapped_function(
    clock, make_async_breaker
):
    breaker = make_async_breaker(clock)
    calls = []

    @breaker.decorate
    async def fail():
        calls.append(1)
        raise ValueError("boom")

    await open_the_circuit(breaker, fail)
    assert len(calls) == 3

    with pytest.raises(CircuitBreakerError):
        await fail()

    assert len(calls) == 3


async def test_async_half_open_after_recovery_lets_one_call_through(
    clock, make_async_breaker
):
    breaker = make_async_breaker(clock)
    calls = []

    @breaker.decorate
    async def sometimes():
        calls.append(1)
        if len(calls) <= 3:
            raise ValueError("boom")
        return "ok"

    await open_the_circuit(breaker, sometimes)

    clock.advance(60)
    assert not breaker.opened

    assert await sometimes() == "ok"
    assert breaker.closed
    assert breaker.failure_count == 0


async def test_async_rejected_exception_is_not_a_failure_but_resets(
    clock, make_async_breaker
):
    breaker = make_async_breaker(
        clock,
        expected_exception=lambda exc_type, exc_value: isinstance(
            exc_value, ValueError
        ),
    )
    raises = {"exc": ValueError}

    @breaker.decorate
    async def operation():
        raise raises["exc"]("boom")

    for _ in range(2):
        with pytest.raises(ValueError):
            await operation()

    assert breaker.failure_count == 2

    raises["exc"] = KeyError
    with pytest.raises(KeyError):
        await operation()

    assert breaker.failure_count == 0
    assert breaker.closed


async def test_decorated_target_is_still_a_coroutine_function(
    clock, make_async_breaker
):
    breaker = make_async_breaker(clock)

    @breaker.decorate
    async def get_files(store_id):
        """List the files in a store."""
        return [store_id]

    assert inspect.iscoroutinefunction(get_files)
    assert get_files.__name__ == "get_files"
    assert get_files.__doc__ == "List the files in a store."
    assert await get_files("hub") == ["hub"]


async def test_decorate_rejects_sync_and_async_generator(clock, make_async_breaker):
    breaker = make_async_breaker(clock)

    def gen():
        yield 1

    async def agen():
        yield 1

    with pytest.raises(ValueError, match="gen"):
        breaker.decorate(gen)

    with pytest.raises(ValueError, match="agen"):
        breaker.decorate(agen)
