# Copyright 2021-2025 Zenauth Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cerbos.sdk._breaker import CircuitBreakerError


def open_the_circuit(breaker, wrapped) -> None:
    for _ in range(3):
        with pytest.raises(ValueError):
            wrapped()

    assert breaker.opened


class Store:
    def get_files(self):
        return []


def test_opens_at_threshold_not_before(clock, make_breaker):
    breaker = make_breaker(clock)

    @breaker.decorate
    def fail():
        raise ValueError("boom")

    for _ in range(2):
        with pytest.raises(ValueError):
            fail()

    assert breaker.closed
    assert breaker.failure_count == 2

    with pytest.raises(ValueError):
        fail()

    assert breaker.opened
    assert breaker.failure_count == 3


def test_success_closes_and_zeroes_count(clock, make_breaker):
    breaker = make_breaker(clock)
    attempts = []

    @breaker.decorate
    def flaky():
        attempts.append(1)
        if len(attempts) <= 2:
            raise ValueError("boom")
        return "ok"

    for _ in range(2):
        with pytest.raises(ValueError):
            flaky()

    assert breaker.failure_count == 2

    assert flaky() == "ok"
    assert breaker.closed
    assert breaker.failure_count == 0
    assert breaker.last_failure is None


def test_rejected_exception_is_not_a_failure_but_resets(clock, make_breaker):
    breaker = make_breaker(
        clock,
        expected_exception=lambda exc_type, exc_value: isinstance(
            exc_value, ValueError
        ),
    )
    raises = {"exc": ValueError}

    @breaker.decorate
    def operation():
        raise raises["exc"]("boom")

    for _ in range(2):
        with pytest.raises(ValueError):
            operation()

    assert breaker.failure_count == 2

    raises["exc"] = KeyError
    with pytest.raises(KeyError):
        operation()

    assert breaker.failure_count == 0
    assert breaker.closed


def test_open_circuit_does_not_call_the_wrapped_function(clock, make_breaker):
    breaker = make_breaker(clock)
    calls = []

    @breaker.decorate
    def fail():
        calls.append(1)
        raise ValueError("boom")

    open_the_circuit(breaker, fail)
    assert len(calls) == 3

    with pytest.raises(CircuitBreakerError):
        fail()

    assert len(calls) == 3


def test_half_open_after_recovery_lets_one_call_through(clock, make_breaker):
    breaker = make_breaker(clock)
    calls = []

    @breaker.decorate
    def sometimes():
        calls.append(1)
        if len(calls) <= 3:
            raise ValueError("boom")
        return "ok"

    open_the_circuit(breaker, sometimes)

    clock.advance(60)
    assert not breaker.opened

    assert sometimes() == "ok"
    assert breaker.closed
    assert breaker.failure_count == 0


def test_failure_while_half_open_reopens_at_once(clock, make_breaker):
    breaker = make_breaker(clock)

    @breaker.decorate
    def fail():
        raise ValueError("boom")

    open_the_circuit(breaker, fail)

    clock.advance(60)
    with pytest.raises(ValueError):
        fail()

    assert breaker.opened
    assert breaker.failure_count == 4, "only a success resets the count"
    assert breaker.open_remaining == 60


def test_half_open_boundary_is_inclusive(clock, make_breaker):
    breaker = make_breaker(clock)

    @breaker.decorate
    def fail():
        raise ValueError("boom")

    open_the_circuit(breaker, fail)

    clock.advance(59)
    assert breaker.opened, "still inside the recovery window"

    clock.advance(1)
    assert not breaker.opened, "at the recovery timeout the breaker lets a call through"


def test_error_message_format(clock, make_breaker):
    breaker = make_breaker(clock, failure_threshold=1, recovery_timeout=30, name="boom")

    @breaker.decorate
    def operation():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        operation()

    clock.advance(10)
    with pytest.raises(CircuitBreakerError) as excinfo:
        operation()

    message = str(excinfo.value)
    assert message.startswith('Circuit "boom" OPEN until ')
    assert message.endswith(
        "(1 failures, 20 sec remaining) (last_failure: ValueError('boom'))"
    )


def test_name_defaults_to_qualname(clock, make_breaker):
    breaker = make_breaker(clock)
    breaker.decorate(Store.get_files)
    assert breaker.name == "Store.get_files"


def test_name_falls_back_to_dunder_name(clock, make_breaker):
    class Legacy:
        __name__ = "legacy"

        def __call__(self):
            return "ok"

    breaker = make_breaker(clock)
    breaker.decorate(Legacy())
    assert breaker.name == "legacy"


def test_unnamed_breaker_has_a_placeholder_name(clock, make_breaker):
    assert make_breaker(clock).name == "unnamed_CircuitBreaker"


def test_wrapper_keeps_metadata(clock, make_breaker):
    breaker = make_breaker(clock)

    @breaker.decorate
    def get_files(store_id):
        """List the files in a store."""
        return [store_id]

    assert get_files.__name__ == "get_files"
    assert get_files.__doc__ == "List the files in a store."
    assert get_files("hub") == ["hub"]


def test_predicate_accepts_class_iterable_and_callable(clock, make_breaker):
    by_class = make_breaker(clock, expected_exception=ValueError)
    assert by_class.is_failure(ValueError, ValueError())
    assert not by_class.is_failure(KeyError, KeyError())

    by_iterable = make_breaker(clock, expected_exception=(ValueError, KeyError))
    assert by_iterable.is_failure(KeyError, KeyError())
    assert not by_iterable.is_failure(TypeError, TypeError())

    by_callable = make_breaker(
        clock,
        expected_exception=lambda exc_type, exc_value: isinstance(exc_value, KeyError),
    )
    assert by_callable.is_failure(KeyError, KeyError())
    assert not by_callable.is_failure(ValueError, ValueError())


def test_predicate_rejects_string(clock, make_breaker):
    with pytest.raises(ValueError, match="Did you mean name?"):
        make_breaker(clock, expected_exception="ValueError")


def test_decorate_rejects_generator_function(clock, make_breaker):
    breaker = make_breaker(clock)

    def gen():
        yield 1

    with pytest.raises(ValueError, match="gen"):
        breaker.decorate(gen)


def test_decorate_rejects_coroutine_function(clock, make_breaker):
    breaker = make_breaker(clock)

    async def operation():
        return "ok"

    with pytest.raises(ValueError, match="operation"):
        breaker.decorate(operation)
