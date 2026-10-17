"""Per-session conversation memory and a readers-writer lock."""

from __future__ import annotations

import threading
from collections import OrderedDict
from contextlib import contextmanager

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class SessionStore:
    """Chat histories keyed by session id, with LRU eviction.

    In-memory, so histories are lost on restart and not shared between
    processes. For multi-instance deployments, swap `InMemoryChatMessageHistory`
    for a persistent `BaseChatMessageHistory` (Redis, Postgres) - the rest of
    the code only uses the base interface.
    """

    def __init__(self, max_sessions: int = 500):
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, InMemoryChatMessageHistory] = OrderedDict()
        self._lock = threading.Lock()

    def _get(self, session_id: str) -> InMemoryChatMessageHistory:
        history = self._sessions.get(session_id)
        if history is None:
            history = InMemoryChatMessageHistory()
            self._sessions[session_id] = history
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
        self._sessions.move_to_end(session_id)
        return history

    def recent(self, session_id: str, max_turns: int) -> list[BaseMessage]:
        with self._lock:
            if session_id not in self._sessions:
                return []
            return list(self._get(session_id).messages[-2 * max_turns :])

    def append(self, session_id: str, question: str, answer: str) -> None:
        with self._lock:
            self._get(session_id).add_messages([HumanMessage(question), AIMessage(answer)])

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def __len__(self) -> int:
        return len(self._sessions)


class ReadWriteLock:
    """Many concurrent readers (chat requests) or one writer (re-ingestion).

    Writer-preferring: once a re-ingest is waiting, new readers queue behind
    it, so a busy API cannot starve ingestion forever.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0

    @contextmanager
    def read(self):
        with self._cond:
            while self._writer or self._writers_waiting:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if not self._readers:
                    self._cond.notify_all()

    @contextmanager
    def write(self):
        with self._cond:
            self._writers_waiting += 1
            while self._writer or self._readers:
                self._cond.wait()
            self._writers_waiting -= 1
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()
