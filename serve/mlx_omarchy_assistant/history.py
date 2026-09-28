"""Opt-in local history and ordered turn events."""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path

MAX_TEXT = 1024 * 1024
MAX_HISTORY = 8 * MAX_TEXT
MAX_PINNED = 64 * 1024
CHECKPOINT_BYTES = 256 * 1024
CHECKPOINT_SECONDS = 2.0
ID = re.compile(r"^[0-9a-f]{32}$")


def _completed_turn_ids(messages):
    users = {m.get("turn_id") for m in messages if m.get("role") == "user" and m.get("status") == "complete"}
    ended = {m.get("turn_id") for m in messages if m.get("role") == "assistant" and m.get("status") != "streaming"}
    return users & ended


def _valid_context_shape(raw):
    if not isinstance(raw, dict):
        return False
    ids = raw.get("selected_turn_ids")
    if ids is not None and (not isinstance(ids, list) or any(not isinstance(turn, str) for turn in ids)):
        return False
    pinned = raw.get("pinned_constraints")
    return isinstance(pinned, str) and len(pinned.encode()) <= MAX_PINNED


class BusyError(ValueError):
    pass


class EventGap(ValueError):
    pass


class ConversationStore:
    def __init__(self, home: Path):
        self.directory = home / "assistant" / "history"
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.records = {}
        self.streams = {}
        self.cancelled = set()
        self.pending = {}
        if self.directory.exists():
            for path in self.directory.glob("*.json"):
                if not ID.fullmatch(path.stem) or path.is_symlink() or path.stat().st_size > MAX_HISTORY:
                    continue
                try:
                    record = json.loads(path.read_text())
                    if record.get("id") != path.stem or not isinstance(record.get("messages"), list):
                        continue
                    for message in record["messages"]:
                        if message.get("status") == "streaming":
                            message["status"] = "stopped"
                    record["active_turn"] = None
                    record["save"] = True
                    record.pop("context_error", None)
                    raw = record.pop("context_raw", record.get("context"))
                    context = {"selected_turn_ids": None, "pinned_constraints": ""}
                    error = None
                    if raw is None and "context" not in record:
                        pass
                    elif not _valid_context_shape(raw):
                        error = "Saved context is malformed; pinned constraints are withheld until repaired"
                    else:
                        pairs = _completed_turn_ids(record["messages"])
                        ids = raw["selected_turn_ids"]
                        if ids is not None and any(turn not in pairs for turn in ids):
                            error = "Saved context selects turns missing from this conversation; selection is withheld until repaired"
                        else:
                            context = {"selected_turn_ids": None if ids is None else list(ids),
                                       "pinned_constraints": raw["pinned_constraints"]}
                    record["context"] = context
                    if error:
                        record["context_error"] = error
                        record["context_raw"] = raw
                    self.records[path.stem] = record
                    self.streams[path.stem] = deque(maxlen=4096)
                    self.pending[path.stem] = [0, time.monotonic()]
                except (OSError, ValueError, TypeError, AttributeError):
                    continue

    def _record(self, cid):
        if not isinstance(cid, str) or not ID.fullmatch(cid):
            raise ValueError("Invalid conversation ID")
        if cid not in self.records:
            raise KeyError("Conversation not found")
        return self.records[cid]

    def _save(self, record):
        if not record["save"]:
            self.pending[record["id"]] = [0, time.monotonic()]
            return
        encoded = json.dumps(record, ensure_ascii=False, allow_nan=False).encode()
        if len(encoded) > MAX_HISTORY:
            raise ValueError("Conversation storage limit reached; export this conversation and start another")
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, temporary = tempfile.mkstemp(dir=self.directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.directory / (record["id"] + ".json"))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self.pending[record["id"]] = [0, time.monotonic()]

    def create(self, save=False):
        if not isinstance(save, bool):
            raise ValueError("save must be a boolean")
        with self.lock:
            cid = uuid.uuid4().hex
            record = {"id": cid, "save": save, "messages": [], "active_turn": None,
                      "sequence": 0, "created_at": time.time(), "title": "New conversation",
                      "context": {"selected_turn_ids": None, "pinned_constraints": ""}}
            self.records[cid] = record
            self.streams[cid] = deque(maxlen=4096)
            self._save(record)
            return copy.deepcopy(record)

    def list(self):
        with self.lock:
            return [dict({k: record[k] for k in ("id", "title", "save", "created_at")},
                         updated_at=record.get("updated_at", record["created_at"]),
                         message_count=len(record["messages"])) for record in self.records.values()]

    def get(self, cid):
        with self.lock:
            return copy.deepcopy(self._record(cid))

    def set_saved(self, cid, save):
        if not isinstance(save, bool):
            raise ValueError("save must be a boolean")
        with self.lock:
            record = self._record(cid)
            record["save"] = save
            if save:
                self._save(record)
            else:
                (self.directory / (cid + ".json")).unlink(missing_ok=True)
            return copy.deepcopy(record)

    def set_context(self, cid, selected_turn_ids, pinned_constraints):
        with self.lock:
            record = self._record(cid)
            if record["active_turn"]:
                raise BusyError("Stop the active response before changing context selection")
            if not isinstance(pinned_constraints, str):
                raise ValueError("pinned_constraints must be a string")
            if len(pinned_constraints.encode()) > MAX_PINNED:
                raise ValueError("Pinned constraints exceed the 64 KiB limit")
            if selected_turn_ids is None:
                turns = None
            else:
                if not isinstance(selected_turn_ids, list):
                    raise ValueError("selected_turn_ids must be null or a list of turn IDs")
                turns = []
                pairs = _completed_turn_ids(record["messages"])
                for turn in selected_turn_ids:
                    if not isinstance(turn, str):
                        raise ValueError("Turn selections must be turn ID strings")
                    if turn in turns:
                        raise ValueError("Turn %s is selected more than once" % turn)
                    if turn not in pairs:
                        raise ValueError("Turn %s is not a completed turn in this conversation" % turn)
                    turns.append(turn)
            record["context"] = {"selected_turn_ids": turns, "pinned_constraints": pinned_constraints}
            record.pop("context_error", None)
            record.pop("context_raw", None)
            self._save(record)
            return copy.deepcopy(record)

    def delete(self, cid):
        with self.lock:
            record = self._record(cid)
            if record["active_turn"]:
                raise BusyError("Stop the active response before deleting this conversation")
            (self.directory / (cid + ".json")).unlink(missing_ok=True)
            del self.records[cid]
            del self.streams[cid]
            self.pending.pop(cid, None)

    def begin(self, cid, text):
        if not isinstance(text, str) or not text.strip() or len(text.encode()) > MAX_TEXT:
            raise ValueError("A message must contain text and use at most 1 MiB")
        with self.lock:
            record = self._record(cid)
            if record["active_turn"]:
                raise BusyError("This conversation already has an active response")
            if len(record["messages"]) >= 400 or sum(len(m["content"].encode()) for m in record["messages"]) + len(text.encode()) > MAX_HISTORY - MAX_TEXT:
                raise ValueError("Conversation limit reached; export it and start another conversation")
            turn = uuid.uuid4().hex
            record["messages"].extend([
                {"role": "user", "content": text, "turn_id": turn, "status": "complete"},
                {"role": "assistant", "content": "", "turn_id": turn, "status": "streaming", "components": []},
            ])
            record["active_turn"] = turn
            if len(record["messages"]) == 2:
                record["title"] = text[:80]
            self._save(record)
            self.emit(cid, turn, "status", {"state": "working"})
            return turn

    def emit(self, cid, turn, kind, data):
        with self.lock:
            record = self._record(cid)
            if record["active_turn"] != turn or turn in self.cancelled:
                return False
            message = record["messages"][-1]
            if kind == "text":
                text = data["text"]
                if len(message["content"].encode()) + len(text.encode()) > MAX_TEXT:
                    raise ValueError("Response exceeds the 1 MiB display limit")
                message["content"] += text
                self.pending[cid][0] += len(text.encode())
            elif kind == "decision":
                self.pending[cid][0] += len(json.dumps(data, ensure_ascii=False, allow_nan=False).encode())
                message["decision"] = copy.deepcopy(data)
            elif kind == "component":
                self.pending[cid][0] += len(json.dumps(data, ensure_ascii=False, allow_nan=False).encode())
                message["components"].append(copy.deepcopy(data))
            record["updated_at"] = time.time()
            record["sequence"] += 1
            event = {"conversation_id": cid, "turn_id": turn, "sequence": record["sequence"],
                     "type": kind, "data": copy.deepcopy(data)}
            self.streams[cid].append(event)
            pending = self.pending[cid]
            if pending[0] >= CHECKPOINT_BYTES or time.monotonic() - pending[1] >= CHECKPOINT_SECONDS:
                self._save(record)
            self.changed.notify_all()
            return True

    def cancel(self, cid, turn):
        with self.lock:
            record = self._record(cid)
            if record["active_turn"] == turn:
                self.cancelled.add(turn)
                self._save(record)
                return True
            return False

    def finish(self, cid, turn, stopped=False):
        with self.lock:
            record = self._record(cid)
            if record["active_turn"] != turn:
                return
            stopped = stopped or turn in self.cancelled
            self.cancelled.discard(turn)
            record["messages"][-1]["status"] = "stopped" if stopped else "complete"
            self.emit(cid, turn, "done", {"stopped": stopped})
            record["active_turn"] = None
            self._save(record)
            self.changed.notify_all()

    def events(self, cid, after):
        with self.lock:
            record = self._record(cid)
            if not isinstance(after, int) or after < 0 or after > record["sequence"]:
                raise ValueError("Invalid event cursor")
            stream = self.streams[cid]
            if (stream and after < stream[0]["sequence"] - 1) or (not stream and after < record["sequence"]):
                raise EventGap("Event cursor expired; reload the conversation, not the inference request")
            return copy.deepcopy([event for event in stream if event["sequence"] > after])
