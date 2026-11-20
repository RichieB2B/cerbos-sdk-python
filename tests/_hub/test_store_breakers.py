# Copyright 2021-2025 Zenauth Ltd.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

import grpc
import pytest

from cerbos.cloud.store.v1 import store_pb2
from cerbos.sdk._async._hub import _store as async_store
from cerbos.sdk._breaker import AsyncCircuitBreaker, CircuitBreaker, CircuitBreakerError
from cerbos.sdk._sync._hub import _store as sync_store
from cerbos.sdk.hub.model import Credentials
from cerbos.sdk.hub.store_model import (
    AbortedError,
    File,
    FileOps,
    TooManyFailuresError,
)
from google.rpc import code_pb2, status_pb2

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


# code_pb2 gives the numeric codes rpc_status.from_call reads back out of a call, while a
# call reports its own code as a grpc.StatusCode member. The numbers are the same.
_GRPC_STATUS_CODE = {code.value[0]: code for code in grpc.StatusCode}


def _rich_rpc_error(code: int) -> grpc.RpcError:
    """A grpc.RpcError that rpc_status.from_call accepts, as a terminated call looks.

    from_call wants the rich status in the trailing metadata under the key gRPC uses on
    the wire, and checks the call's own code and details against that status.
    """

    status = status_pb2.Status(code=code, message="the store is busy")
    trailing_metadata = (("grpc-status-details-bin", status.SerializeToString()),)

    class RichStatusError(grpc.RpcError):
        def code(self) -> grpc.StatusCode:
            return _GRPC_STATUS_CODE[code]

        def details(self) -> str:
            return status.message

        def initial_metadata(self) -> tuple:
            return ()

        def trailing_metadata(self) -> tuple:
            return trailing_metadata

    return RichStatusError()


def _bare_rpc_error() -> grpc.RpcError:
    """A gRPC error carrying no rich status, which is what a plain transport failure is."""

    class BareStatusError(grpc.RpcError):
        def code(self) -> grpc.StatusCode:
            return grpc.StatusCode.UNAVAILABLE

        def details(self) -> str:
            return "failed to connect to server"

        def initial_metadata(self) -> tuple:
            return ()

        def trailing_metadata(self) -> None:
            return None

    return BareStatusError()


class _StoreStub:
    """Stand-in for CerbosStoreServiceStub: the shared-circuit RPCs fail, ListFiles works."""

    def __init__(self, code: int = code_pb2.ABORTED) -> None:
        self.code = code
        self.calls: list[str] = []

    def _fail(self, rpc: str) -> None:
        self.calls.append(rpc)
        raise _rich_rpc_error(self.code)

    def _succeed(self, rpc: str) -> store_pb2.ListFilesResponse:
        self.calls.append(rpc)
        return store_pb2.ListFilesResponse(store_version=1, files=["example.yaml"])

    def GetFiles(self, *args, **kwargs) -> None:
        self._fail("GetFiles")

    def ModifyFiles(self, *args, **kwargs) -> None:
        self._fail("ModifyFiles")

    def ReplaceFiles(self, *args, **kwargs) -> None:
        self._fail("ReplaceFiles")

    def ListFiles(self, *args, **kwargs) -> store_pb2.ListFilesResponse:
        return self._succeed("ListFiles")


class _AsyncStoreStub(_StoreStub):
    async def GetFiles(self, *args, **kwargs) -> None:
        self._fail("GetFiles")

    async def ModifyFiles(self, *args, **kwargs) -> None:
        self._fail("ModifyFiles")

    async def ReplaceFiles(self, *args, **kwargs) -> None:
        self._fail("ReplaceFiles")

    async def ListFiles(self, *args, **kwargs) -> store_pb2.ListFilesResponse:
        return self._succeed("ListFiles")


def _offline_credentials(monkeypatch) -> Credentials:
    """Credentials for a client whose requests never leave this process.

    The auth client reads the environment before it looks at the credentials it is
    handed, so both are filled in with dummies; the stubbed stub answers every call.
    """

    monkeypatch.setenv("CERBOS_HUB_CLIENT_ID", "test-client")
    monkeypatch.setenv("CERBOS_HUB_CLIENT_SECRET", "test-secret")
    return Credentials("test-client", "test-secret")


def _snapshot_breaker(module, monkeypatch) -> CircuitBreaker:
    """Hand back the module-level circuit, restored to its current state afterwards.

    The store's breakers are process-wide, so a circuit left open by one test would
    refuse the calls of whatever runs next in the session -- including the credentialed
    Hub tests. Re-setting each attribute to the value it holds now is how monkeypatch
    learns what to put back.
    """

    breaker = module._CIRCUIT_BREAKER.__self__
    for attr in ("_state", "_opened", "_failure_count", "_last_failure"):
        monkeypatch.setattr(breaker, attr, getattr(breaker, attr))

    return breaker


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


_NOTABLE_FAILURE_CODES = [
    code_pb2.ABORTED,
    code_pb2.CANCELLED,
    code_pb2.DEADLINE_EXCEEDED,
]
_OTHER_FAILURE_CODES = [
    code_pb2.INVALID_ARGUMENT,
    code_pb2.UNAVAILABLE,
    code_pb2.NOT_FOUND,
]


@pytest.mark.parametrize("module, client, breaker_class", _STORES)
@pytest.mark.parametrize("code", _NOTABLE_FAILURE_CODES)
def test_the_notable_grpc_codes_count_as_failures(module, client, breaker_class, code):
    shared, _ = _breakers(module)
    error = _rich_rpc_error(code)

    assert shared.is_failure(type(error), error), code_pb2.Name(code)


@pytest.mark.parametrize("module, client, breaker_class", _STORES)
@pytest.mark.parametrize("code", _OTHER_FAILURE_CODES)
def test_other_grpc_codes_are_not_failures(module, client, breaker_class, code):
    shared, _ = _breakers(module)
    error = _rich_rpc_error(code)

    assert not shared.is_failure(type(error), error), code_pb2.Name(code)


@pytest.mark.parametrize("module, client, breaker_class", _STORES)
def test_only_the_notable_grpc_failures_count(module, client, breaker_class):
    shared, _ = _breakers(module)

    # Neither an exception from outside gRPC nor a gRPC error without a rich status is a
    # failure; the library default of counting every Exception would have counted both.
    assert not shared.is_failure(RuntimeError, RuntimeError("not a gRPC error"))
    bare = _bare_rpc_error()
    assert not shared.is_failure(type(bare), bare)


@pytest.mark.anyio
async def test_grpc_aborteds_open_the_shared_circuit(monkeypatch):
    breaker = _snapshot_breaker(async_store, monkeypatch)
    stub = _AsyncStoreStub()

    async with async_store.AsyncCerbosHubStoreClient(
        _offline_credentials(monkeypatch)
    ) as client:
        client._store_stub = stub

        for attempt in range(1, 10):
            with pytest.raises(AbortedError):
                await client.get_files("hub", ["example.yaml"])

            assert not breaker.opened, f"the circuit opened after {attempt} failures"

        with pytest.raises(AbortedError):
            await client.get_files("hub", ["example.yaml"])

        assert breaker.opened
        assert breaker.failure_count == 10

        # The tenth failure opened the circuit these methods share, so modify_files is
        # refused as too-many-failures without reaching the store.
        with pytest.raises(TooManyFailuresError):
            await client.modify_files(
                "hub", "message", FileOps(add=[File("example.yaml", b"hello")])
            )

        assert stub.calls == ["GetFiles"] * 10

        # list_files has a circuit of its own and still works.
        assert (await client.list_files("hub")).files() == ["example.yaml"]
        assert stub.calls == ["GetFiles"] * 10 + ["ListFiles"]


def test_grpc_aborteds_open_the_shared_circuit_sync(monkeypatch):
    breaker = _snapshot_breaker(sync_store, monkeypatch)
    stub = _StoreStub()

    with sync_store.CerbosHubStoreClient(_offline_credentials(monkeypatch)) as client:
        client._store_stub = stub

        for attempt in range(1, 10):
            with pytest.raises(AbortedError):
                client.get_files("hub", ["example.yaml"])

            assert not breaker.opened, f"the circuit opened after {attempt} failures"

        with pytest.raises(AbortedError):
            client.get_files("hub", ["example.yaml"])

        assert breaker.opened
        assert breaker.failure_count == 10

        with pytest.raises(TooManyFailuresError):
            client.modify_files(
                "hub", "message", FileOps(add=[File("example.yaml", b"hello")])
            )

        assert stub.calls == ["GetFiles"] * 10

        assert client.list_files("hub").files() == ["example.yaml"]
        assert stub.calls == ["GetFiles"] * 10 + ["ListFiles"]


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
