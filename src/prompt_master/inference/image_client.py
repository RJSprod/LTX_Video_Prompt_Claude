"""The application's llama-server client, in the shape the image engine wants.

``image_engine`` declares the slice of a chat client it needs as a structural
protocol and imports nothing from this package, which is what lets its 157
tests run with no Qt, no server and no network. The real ``LlamaClient``
predates it and does not match, so the difference is absorbed here rather than
in the engine — the spec is explicit that adapting is the supported move and
editing the engine is not.

Two things differ, and only one of them is interesting.

*The message shape.* The engine hands over a system prompt, a user turn and an
optional still. ``LlamaClient`` wants the OpenAI list, with the image part
ahead of the text part. That is a rewrite of two dictionaries.

*The direction.* ``LlamaClient`` pushes: it calls a callback per fragment and
returns the whole reply when the response ends. The engine pulls: it iterates
what ``stream_chat`` yields. A thread and a queue bridge the two, and that is
worth the twenty lines it costs, because the alternative — handing the UI the
callback instead — leaks. The engine makes three kinds of call on this client:
the writer pass, whose fragments belong in the output pane, and the
Choose-for-me and smart-negative passes, whose fragments are JSON and a list of
faults and belong nowhere near it. Only the writer pass is yielded back out of
``ImagePromptEngine.stream``, so pulling is what keeps the other two out of the
pane.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from typing import Any

from .llama_client import LlamaClient

# Enough for the longest writer pass and no more. The widest word band any
# profile declares is 30-140 words with a 300-word hard ceiling, and the other
# two passes return a short JSON object and a dozen noun phrases. A ceiling
# well above the ask still bounds a model that decides not to stop.
MAX_TOKENS = 900

# The end of one reply. A sentinel rather than a closed queue because the
# producer has to say "finished" and "finished badly" through the same channel.
_DONE = object()


class _Failed:
    """Carries an exception across the queue so the consumer can re-raise it."""

    __slots__ = ("error",)

    def __init__(self, error: BaseException) -> None:
        self.error = error


class ImageChatClient:
    """``LlamaClient`` as ``image_engine.ChatClient``.

    One instance per generation. It holds the cancel flag the window's Cancel
    button sets, so a stopped generation stops the request rather than only the
    display of it.
    """

    def __init__(self, client: LlamaClient, cancel: threading.Event | None = None,
                 max_tokens: int = MAX_TOKENS) -> None:
        self.client, self.cancel, self.max_tokens = client, cancel, max_tokens

    def stream_chat(self, *, system: str, user: str, temperature: float, top_p: float,
                    seed: int, image_jpeg_b64: str | None = None) -> Iterator[str]:
        """Yield the reply in the fragments it arrives in.

        The request runs on a thread of its own and posts fragments into a
        queue; this generator drains it. An exception on that thread is carried
        across and raised here, on the thread that asked for the reply, so the
        engine's own error handling — the selection pass falls back to keyword
        matching, the refine pass gives up and keeps the banks — sees it.
        """
        messages = _messages(system, user, image_jpeg_b64)
        fragments: queue.Queue[Any] = queue.Queue()

        def pump() -> None:
            try:
                self.client.stream_chat(messages, self.max_tokens, seed, fragments.put,
                                        self.cancel, temperature=temperature, top_p=top_p)
            except BaseException as error:            # noqa: BLE001 - re-raised below
                fragments.put(_Failed(error))
            finally:
                fragments.put(_DONE)

        thread = threading.Thread(target=pump, name="image-writer", daemon=True)
        thread.start()
        while True:
            item = fragments.get()
            if item is _DONE:
                break
            if isinstance(item, _Failed):
                raise item.error
            yield item
        # Only on the way out of a reply that finished: an abandoned generator
        # leaves a daemon thread to end on its own, and joining one that is
        # still inside a 600-second read would hang the generation being
        # cancelled.
        thread.join(timeout=5)


def _messages(system: str, user: str, image: str | None) -> list[dict[str, Any]]:
    """The two turns, with the still ahead of the words when there is one.

    Same order as prompt mode's, which is upstream's: llama.cpp resolves the
    image part against the text that follows it, and a still after the question
    is a still the question could not have been about.
    """
    content: Any = user
    if image:
        content = [{"type": "image_url", "image_url": {"url": _data_url(image)}},
                   {"type": "text", "text": user}]
    return [{"role": "system", "content": system},
            {"role": "user", "content": content}]


def _data_url(image: str) -> str:
    """Accept either of the two forms this string arrives in.

    The engine's protocol names the parameter ``image_jpeg_b64`` and the
    application's ``imaging.preprocess`` produces a full data URL. Rather than
    have one of them convert on every call, both are understood here.
    """
    return image if image.startswith("data:") else f"data:image/jpeg;base64,{image}"
