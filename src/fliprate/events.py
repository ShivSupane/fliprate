"""Progress events. The library yields them; the API forwards them; the console draws them."""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, Generator, Optional

Event = Dict[str, Any]


def make_event(stage: str, **data: Any) -> Event:
    return {"stage": stage, "t": round(time.time(), 3), **data}


def to_sse(event: Event) -> str:
    """Format one event as a Server-Sent Events message."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def drain(gen: Generator, on_event: Optional[Callable[[Event], None]] = None):
    """Run an event generator to the end, optionally calling `on_event`; return its result."""
    while True:
        try:
            ev = next(gen)
        except StopIteration as stop:
            return stop.value
        if on_event:
            on_event(ev)
