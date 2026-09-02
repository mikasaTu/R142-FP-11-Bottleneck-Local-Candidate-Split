#!/usr/bin/env python3
"""Small opt-in bridge for the pinned Evo-1 WebSocket server.

The pinned ``Evo_1/scripts/Evo1_server.py`` treats every message as an
inference payload.  Stage-S must therefore deploy this bridge in the server
process and dispatch control messages before calling the unchanged
``infer_from_json_dict`` function.  This file deliberately does not import,
patch, or rewrite the released Evo source checkout.

Typical integration inside the pinned ``handle_request`` loop is:

    control = dispatcher.control_response(json_data)
    if control is not None:
        await websocket.send(json.dumps(control))
        continue
    actions = infer_from_json_dict(json_data, model, normalizer, arm_key, dataset_key)

The inference branch remains the released implementation.  The wrapper is
fail-closed: a real server must require Torch/CUDA RNG support, and a client
receives an explicit error for malformed or unsupported controls.
"""

from __future__ import annotations

import json
import asyncio
from typing import Any, Callable, Dict, Optional

from r142_stage_s.robotwin import (
    EVO_EXACT_REPLAY_PROTOCOL,
    EvoExactReplayServerControl,
)


class EvoServerReplayDispatcher:
    """Dispatch Stage-S controls while preserving Evo inference semantics."""

    protocol = EVO_EXACT_REPLAY_PROTOCOL

    def __init__(self, *, require_torch: bool = True):
        self.control = EvoExactReplayServerControl(require_torch=require_torch)

    def control_response(self, message: Any) -> Optional[Dict[str, Any]]:
        """Return a response for a control message, else ``None`` for inference."""

        response = self.control.handle_message(message)
        if response is None:
            return None
        return dict(response)

    async def dispatch(self, websocket: Any, message: Any) -> bool:
        """Send a control response and return whether the message was consumed."""

        response = self.control_response(message)
        if response is None:
            return False
        await websocket.send(json.dumps(response, sort_keys=True, separators=(",", ":")))
        return True


def build_handle_request(
    *,
    infer_from_json_dict: Callable[..., Any],
    model: Any,
    normalizer: Any,
    arm_key: str,
    dataset_key: str,
    require_torch: bool = True,
):
    """Build a drop-in async handler around pinned ``infer_from_json_dict``.

    This is the equivalent in-process integration for deployments where the
    released checkout is immutable.  The only added branch handles the three
    versioned controls; the ordinary request is passed to the exact pinned
    inference function with all arguments unchanged.  A shared lock keeps the
    process-global Torch/CUDA streams from being advanced by two concurrent
    WebSocket handlers at once.  Stage-S still deploys one server per rank and
    must not use an external caller that bypasses this handler.
    """

    dispatcher = EvoServerReplayDispatcher(require_torch=require_torch)
    inference_lock = asyncio.Lock()

    async def handle_request(websocket: Any) -> None:
        async for message in websocket:
            json_data = json.loads(message)
            async with inference_lock:
                if await dispatcher.dispatch(websocket, json_data):
                    continue
                actions = infer_from_json_dict(
                    json_data, model, normalizer, arm_key, dataset_key
                )
                await websocket.send(json.dumps(actions))

    return handle_request


def install_dispatcher(*, require_torch: bool = True) -> EvoServerReplayDispatcher:
    """Construct the explicit server-side bridge used by a real deployment."""

    return EvoServerReplayDispatcher(require_torch=require_torch)


__all__ = [
    "EvoServerReplayDispatcher",
    "EVO_EXACT_REPLAY_PROTOCOL",
    "build_handle_request",
    "install_dispatcher",
]
