# Copyright 2021-2025 Zenauth Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Circuit breaking for the Cerbos Hub store client.

A port of the part of the `circuitbreaker` library (2.1.3) that this SDK uses.
That library detects coroutine functions with `asyncio.iscoroutinefunction`,
which Python 3.14 deprecates and Python 3.16 removes, and it is not being
maintained; the state machine lives here instead. The pieces the SDK never used
are left behind: fallback functions, generator decoration and the global
registry of circuits.
"""

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from functools import wraps
from inspect import isclass, iscoroutinefunction, isgeneratorfunction
from math import ceil, floor
from time import monotonic
from typing import Any

_STRING_TYPES = (bytes, str)

STATE_CLOSED = "closed"
STATE_OPEN = "open"
STATE_HALF_OPEN = "half-open"


def _in_exception_list(*exc_types: type[BaseException]) -> Callable[..., bool]:
    """Build a predicate that matches a thrown type against the given exception types."""

    def matches_types(thrown_type: type[BaseException], _: BaseException) -> bool:
        return issubclass(thrown_type, exc_types)

    return matches_types


def _build_failure_predicate(expected_exception: Any) -> Callable[..., bool]:
    """Build the `(type, value) -> bool` predicate that decides what counts as a failure.

    `expected_exception` is an exception type, an iterable of exception types, or a
    predicate function to use as it stands.
    """

    if isclass(expected_exception) and issubclass(expected_exception, Exception):
        return _in_exception_list(expected_exception)

    try:
        iter(expected_exception)
    except TypeError:
        if not callable(expected_exception) or isclass(expected_exception):
            raise ValueError("expected_exception does not look like a predicate")
        return expected_exception

    if isinstance(expected_exception, _STRING_TYPES):
        # ValueError, not TypeError: both the type and this message are the ones the ported
        # library raised for this mistake.
        raise ValueError(  # noqa: TRY004
            "expected_exception cannot be a string. Did you mean name?"
        )

    return _in_exception_list(*expected_exception)


class CircuitBreakerError(Exception):
    """Raised when a call is refused because the circuit is open."""

    def __init__(self, circuit_breaker: "CircuitBreaker") -> None:
        super().__init__()
        self._circuit_breaker = circuit_breaker

    def __str__(self) -> str:
        breaker = self._circuit_breaker
        return (
            f'Circuit "{breaker.name}" OPEN until {breaker.open_until} '
            f"({breaker.failure_count} failures, {round(breaker.open_remaining)} sec remaining) "
            f"(last_failure: {breaker.last_failure!r})"
        )


class CircuitBreaker:
    """Counts matching failures and refuses calls until the recovery timeout passes.

    The circuit closes while calls succeed, opens once `failure_threshold` matching
    failures have piled up, and reports itself half-open as soon as `recovery_timeout`
    seconds have elapsed, letting the next call through: success closes it, a failure
    opens it again and restarts the timeout.
    """

    FAILURE_THRESHOLD = 5
    RECOVERY_TIMEOUT = 30
    EXPECTED_EXCEPTION: Any = Exception

    def __init__(
        self,
        failure_threshold: int | None = None,
        recovery_timeout: float | None = None,
        expected_exception: Any = None,
        name: str | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        """
        :param failure_threshold: break open after this many failures
        :param recovery_timeout: let a call through after this many seconds
        :param expected_exception: an exception type, an iterable of exception types,
            or a `(type, value) -> bool` predicate
        :param name: name for this circuit breaker, defaulting to the decorated function's
        :param clock: source of the elapsed-time readings, so tests can move time
        """

        self._failure_count = 0
        self._last_failure: BaseException | None = None
        self._failure_threshold = failure_threshold or self.FAILURE_THRESHOLD
        self._recovery_timeout = recovery_timeout or self.RECOVERY_TIMEOUT

        # In order of precedence, prefer the constructor argument, then the EXPECTED_EXCEPTION
        # attribute of the concrete type, then this class's. The __dict__ lookup avoids
        # descriptor magic turning a predicate function set by a subclass into a method.
        if not expected_exception:
            try:
                expected_exception = type(self).__dict__["EXPECTED_EXCEPTION"]
            except KeyError:
                expected_exception = CircuitBreaker.EXPECTED_EXCEPTION

        self._failure_predicate = _build_failure_predicate(expected_exception)
        self._name = name
        self._clock = clock
        self._state = STATE_CLOSED
        self._opened = clock()

    def is_failure(
        self, exc_type: type[BaseException], exc_value: BaseException
    ) -> bool:
        """Report whether a thrown exception counts as a failure for this circuit."""

        return self._failure_predicate(exc_type, exc_value)

    def reset(self) -> None:
        """Close the circuit and forget the failures counted so far."""

        self._state = STATE_CLOSED
        self._last_failure = None
        self._failure_count = 0

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc_value, _traceback) -> bool:
        if exc_type and self.is_failure(exc_type, exc_value):
            self._last_failure = exc_value
            self._call_failed()
        else:
            self.reset()

        return False  # False so that an exception, if any, keeps propagating

    def decorate(self, function: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap `function` so that its failures drive this circuit."""

        self._check_function(function)

        if self._name is None:
            try:
                self._name = function.__qualname__
            except AttributeError:
                self._name = function.__name__

        return self._wrap(function)

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call `func`, recording the outcome on this circuit."""

        with self:
            return func(*args, **kwargs)

    def _check_function(self, function: Callable[..., Any]) -> None:
        """Reject a function this circuit cannot observe failing."""

        if isgeneratorfunction(function):
            raise ValueError(
                f"CircuitBreaker cannot wrap the generator function "
                f"{getattr(function, '__qualname__', function)}: wrapping observes the "
                f"generator's creation, never its body. Use AsyncCircuitBreaker for an "
                f"'async def' function."
            )

        if iscoroutinefunction(function):
            raise ValueError(
                f"CircuitBreaker cannot wrap the coroutine function "
                f"{getattr(function, '__qualname__', function)}: it would observe the "
                f"coroutine's creation, never its completion. Use AsyncCircuitBreaker."
            )

    def _wrap(self, function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if self.opened:
                raise CircuitBreakerError(self)

            return self.call(function, *args, **kwargs)

        return wrapper

    def _call_failed(self) -> None:
        self._failure_count += 1

        if self._failure_count >= self._failure_threshold:
            self._state = STATE_OPEN
            self._opened = self._clock()

    @property
    def state(self) -> str:
        if self._state == STATE_OPEN and self.open_remaining <= 0:
            return STATE_HALF_OPEN

        return self._state

    @property
    def open_until(self) -> datetime:
        """The approximate moment the circuit will try to recover."""

        return datetime.now(timezone.utc) + timedelta(seconds=self.open_remaining)

    @property
    def open_remaining(self) -> int:
        """Seconds left before the circuit stops refusing calls."""

        remain = (self._opened + self._recovery_timeout) - self._clock()
        return ceil(remain) if remain > 0 else floor(remain)

    @property
    def failure_count(self) -> int:
        return self._failure_count

    @property
    def failure_threshold(self) -> int:
        return self._failure_threshold

    @property
    def recovery_timeout(self) -> int | float:
        return self._recovery_timeout

    @property
    def last_failure(self) -> BaseException | None:
        return self._last_failure

    @property
    def closed(self) -> bool:
        return self.state == STATE_CLOSED

    @property
    def opened(self) -> bool:
        return self.state == STATE_OPEN

    @property
    def name(self) -> str:
        return self._name or "unnamed_CircuitBreaker"
