"""Whole messages off RLBot's socket: a message that arrives in two pieces while the socket is polled without blocking
is read whole (rlbot before 2.0.0b56 lost the first piece and the stream's framing), and the fix is applied only to
those versions."""

import pytest

from cds_interpret import _rlbot as R


class FakeSocket:
    """A socket whose bytes arrive in pieces: ``now`` has arrived; ``later`` arrive one piece each time a blocking
    read finds nothing waiting. Polled without blocking with nothing waiting, it raises BlockingIOError."""

    def __init__(self, now: bytes, later: list[bytes]):
        self.buf = bytearray(now)
        self.later = list(later)
        self.timeout: float | None = 0.0                # non-blocking, as RLBot's managers poll

    def gettimeout(self):
        return self.timeout

    def close(self):
        pass

    def settimeout(self, t):
        self.timeout = t

    def setblocking(self, flag):
        self.timeout = None if flag else 0.0

    def getblocking(self):
        return self.timeout != 0.0

    def _wait(self) -> None:
        if not self.buf:
            if self.timeout == 0.0:
                raise BlockingIOError
            if self.later:
                self.buf += self.later.pop(0)

    def recv(self, n):
        self._wait()
        out = bytes(self.buf[:n])
        del self.buf[:n]
        return out

    def recv_into(self, view):
        self._wait()
        n = min(len(view), len(self.buf))
        view[:n] = self.buf[:n]
        del self.buf[:n]
        return n


class Relay:
    """rlbot's SocketRelay reading, as in rlbot 2.0.0b54: a 2-byte big-endian length, then the message."""

    def __init__(self, sock):
        self.socket = sock

    def _read_exact(self, n):
        buff = bytearray(n)
        view = memoryview(buff)
        pos = 0
        while pos < n:
            cr = self.socket.recv_into(view[pos:])
            if cr == 0:
                raise EOFError
            pos += cr
        return bytes(buff)

    def read_message(self):                              # the old way
        size = int.from_bytes(self._read_exact(2), "big")
        return self._read_exact(size)


def stream() -> FakeSocket:
    # "hello" (5 bytes), then "hi": only the first byte of the length has arrived when the bot polls
    return FakeSocket(b"\x00", [b"\x05hel", b"lo\x00\x02hi"])


def test_a_message_in_two_pieces_is_read_whole_and_the_stream_stays_framed():
    sock = stream()
    relay = Relay(sock)
    assert R.read_message_whole(relay) == b"hello"
    assert sock.timeout == 0.0                           # the caller's non-blocking mode is back
    assert R.read_message_whole(relay) == b"hi"
    with pytest.raises(BlockingIOError):                 # nothing waiting: as before, for the caller to handle
        R.read_message_whole(relay)
    sock.settimeout(None)
    with pytest.raises(EOFError):                        # the server went away
        R.read_message_whole(relay)


def test_the_old_way_loses_its_place():
    relay = Relay(stream())
    with pytest.raises(BlockingIOError):                 # half a length: the first byte is read, then lost
        relay.read_message()
    relay.socket.settimeout(None)
    with pytest.raises(EOFError):                        # framing gone: bytes 5 and 104 read as a length of 1,384
        relay.read_message()


def test_versions():
    assert R.needs_whole_messages("2.0.0-beta.54") and R.needs_whole_messages("2.0.0b55")
    assert not R.needs_whole_messages("2.0.0b56") and not R.needs_whole_messages("2.0.0")
    assert not R.needs_whole_messages("2.1.0b1") and not R.needs_whole_messages("not a version")


def test_fix_applies_to_older_rlbot_only(monkeypatch):
    interface = pytest.importorskip("rlbot.interface")
    import rlbot.version
    relay_cls = interface.SocketRelay
    original = getattr(relay_cls, "_cds_original_read_message", relay_cls.read_message)
    monkeypatch.setattr(relay_cls, "read_message", original)
    monkeypatch.setattr(rlbot.version, "__version__", "2.0.0b56")
    assert R.fix_socket_reads() is False and relay_cls.read_message is original
    monkeypatch.setattr(rlbot.version, "__version__", "2.0.0-beta.54")
    assert R.fix_socket_reads() is True and R.fix_socket_reads() is True
    relay = relay_cls("test")
    relay.socket = stream()
    assert relay.read_message() == b"hello" and relay.read_message() == b"hi"
