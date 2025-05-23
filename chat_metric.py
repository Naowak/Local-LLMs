"""
title: Chat Metrics
author: constLiakos
funding_url: https://github.com/open-webui
version: 0.0.4
license: MIT
changelog:
- 0.0.1 - Initial upload to openwebui community.
- 0.0.2 - format, remove unnecessary code
- 0.0.3 - add advanced stats: tokens & elapsed time
- 0.0.4 - each metric has its enable switch Valve, experimental metrics are disabled by default
"""

from pydantic import BaseModel, Field
from typing import Optional, Callable, Any, Awaitable
import tiktoken
import os
import time
from utils.misc import get_last_assistant_message


def num_tokens_from_string(user_message: str, model_name: str) -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    print(encoding)
    num_tokens = len(encoding.encode(user_message))
    return num_tokens


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=5, description="Priority level for the filter operations."
        )
        elapsed_time: bool = Field(
            default=True,
            description="Enable for advanced stats",
        )
        tokens_no: bool = Field(
            default=False,
            description="Display total Tokens (Experimantal, NOT Accurate)",
        )
        tokens_per_sec: bool = Field(
            default=False,
            description="Display Tokens per Second (Experimantal, NOT Accurate)",
        )
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.start_time = None
        pass

    def inlet(
        self,
        body: dict,
    ):
        self.start_time = time.time()
        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __model__: Optional[dict] = None,
    ) -> dict:
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        elapsed_time_str = f"Elapsed time: {elapsed_time:.2f} seconds"
        response_message = get_last_assistant_message(body["messages"])
        # model = __model__["id"]
        model = "gpt-4o"
        tokens = num_tokens_from_string(response_message, model)
        tokens_per_sec = tokens / elapsed_time
        stats_array = []

        if self.valves.tokens_per_sec:
            stats_array.append(f"{tokens_per_sec:.2f} T/s")
        if self.valves.tokens_no:
            stats_array.append(f"{tokens} tokens")
        if self.valves.elapsed_time:
            stats_array.append(f"{elapsed_time:.2f} sec")

        stats = " | ".join(stat for stat in stats_array)
        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": stats,
                    "done": True,
                },
            }
        )
        return body
