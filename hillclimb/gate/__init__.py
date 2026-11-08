"""USB gate: decides when the left arm may touch blocks (SPEC section 7).

The hillclimb server uses exactly these two:
  http_handler(method, path, body) -> (status, dict)   routes /gate/*
  start()                                              starts the frame and jaw pollers once, at server start
"""
from __future__ import annotations


def http_handler(method: str, path: str, body=None):
    from .service import handle

    return handle(method, path, body if isinstance(body, dict) else {})


def start() -> None:
    from .service import get_gate

    get_gate()
