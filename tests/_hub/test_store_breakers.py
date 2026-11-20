# Copyright 2021-2025 Zenauth Ltd.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

import pytest

from cerbos.sdk._async._hub import _store as async_store
from cerbos.sdk._breaker import AsyncCircuitBreaker, CircuitBreaker, CircuitBreakerError
from cerbos.sdk._sync._hub import _store as sync_store
from cerbos.sdk.hub.store_model import TooManyFailuresError

# The store's five write/read methods are decorated by a single shared breaker, while
# list_files has its own. That is the behaviour the port preserves, so it needs a test.
_SHARED_BREAKER_METHODS = [
    "replace_files",
    "replace_files_lenient",
    "modify_files",
    "modify_files_lenient",
    "get_files",
]

_STORES = [
    pytest.param(
        async_store,
        async_store.AsyncCerbosHubStoreClient,
        AsyncCircuitBreaker,
        id="async",
    ),
    pytest.param(
        sync_store,
        sync_store.CerbosHubStoreClient,
        CircuitBreaker,
        id="sync",
    ),
]


def _breakers(module) -> tuple[CircuitBreaker, CircuitBreaker]:
    return module._CIRCUIT_BREAKER.__self__, module._LIST_FILES_CIRCUIT_BREAKER.__self__


def _breaker_of(method) -> CircuitBreaker:
    """The breaker a decorated method's wrapper was built with.

    Each store method is wrapped twice, by `handle_store_errors` over the breaker, and
    `functools.wraps` chains them with `__wrapped__`.
    """

    original = method
    while method is not None:
        for cell in getattr(method, "__closure__", None) or ():
            try:
                contents = cell.cell_contents
            except ValueError:  # a free variable that is not bound yet
                continue

            if isinstance(contents, CircuitBreaker):
                return contents

        method = getattr(method, "__wrapped__", None)

    raise AssertionError(f"{original} is not wrapped by a circuit breaker")


def test_importing_the_hub_store_emits_no_deprecation_warnings():
    # Importing the store decorates its methods, which is where the
    # `asyncio.iscoroutinefunction` DeprecationWarning used to come from. It has to be
    # a fresh interpreter: the modules are already imported in this test session.
    code = (
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import cerbos.sdk.hub.store\n"
        "print([str(w.message) for w in caught\n"
        "       if issubclass(w.category, DeprecationWarning)])\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,  # the assertion below reports the child's stderr
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", result.stdout


@pytest.mark.parametrize("module, client, breaker_class", _STORES)
def test_shared_breaker_tuning_and_name(module, client, breaker_class):
    shared, list_files_breaker = _breakers(module)

    assert type(shared) is breaker_class
    assert type(list_files_breaker) is breaker_class
    assert shared is not list_files_breaker

    for breaker in (shared, list_files_breaker):
        assert breaker.failure_threshold == 10
        assert breaker.recovery_timeout == 60

    # Named after the first method it decorated, as before.
    assert shared.name.endswith("replace_files")
    assert list_files_breaker.name.endswith("list_files")


@pytest.mark.parametrize("module, client, breaker_class", _STORES)
def test_five_methods_share_one_breaker_and_list_files_has_its_own(
    module, client, breaker_class
):
    shared, list_files_breaker = _breakers(module)

    for method_name in _SHARED_BREAKER_METHODS:
        assert _breaker_of(getattr(client, method_name)) is shared, method_name

    assert _breaker_of(client.list_files) is list_files_breaker


@pytest.mark.parametrize("module, client, breaker_class", _STORES)
def test_only_the_notable_grpc_failures_count(module, client, breaker_class):
    shared, _ = _breakers(module)

    # The store's predicate counts only ABORTED, CANCELLED and DEADLINE_EXCEEDED; the
    # library default would have counted this one.
    assert not shared.is_failure(RuntimeError, RuntimeError("not a gRPC error"))


@pytest.mark.anyio
async def test_breaker_error_maps_to_too_many_failures():
    breaker = AsyncCircuitBreaker(name="hub")

    @async_store.handle_store_errors
    async def get_files():
        raise CircuitBreakerError(breaker)

    with pytest.raises(TooManyFailuresError) as excinfo:
        await get_files()

    assert str(excinfo.value).startswith("RPC error: TOO_MANY_FAILURES")
    assert str(excinfo.value.underlying).startswith('Circuit "hub" OPEN until ')


def test_breaker_error_maps_to_too_many_failures_sync():
    breaker = CircuitBreaker(name="hub")

    @sync_store.handle_store_errors
    def get_files():
        raise CircuitBreakerError(breaker)

    with pytest.raises(TooManyFailuresError) as excinfo:
        get_files()

    assert str(excinfo.value).startswith("RPC error: TOO_MANY_FAILURES")
    assert str(excinfo.value.underlying).startswith('Circuit "hub" OPEN until ')
