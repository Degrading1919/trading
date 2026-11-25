"""Record the original premarket feed with system receipt timestamps."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from dayrank.alpaca import Alpaca
from dayrank.data import Bar, timestamp
from dayrank.discovery import Window
from dayrank.universe import directory_rows


@asynccontextmanager
async def subscribed_bars(client: Alpaca, feed: str):
    """Verify wildcard subscriptions before trusting a stream."""
    try:
        async with connect(
            f"wss://stream.data.alpaca.markets/v2/{feed}",
            additional_headers=client.websocket_headers(),
            open_timeout=12,
            max_size=8_000_000,
        ) as socket:
            async with asyncio.timeout(12):
                while True:
                    messages = json.loads(await socket.recv())
                    for message in messages:
                        if message.get("T") == "error":
                            raise ValueError(
                                f"Alpaca stream rejected login ({message['code']})"
                            )
                    if any(
                        message.get("msg") == "authenticated" for message in messages
                    ):
                        break
                await socket.send(
                    json.dumps(
                        {
                            "action": "subscribe",
                            "bars": ["*"],
                            "updatedBars": ["*"],
                        }
                    )
                )
                while True:
                    messages = json.loads(await socket.recv())
                    if any(message.get("T") == "error" for message in messages):
                        raise ValueError(
                            "Alpaca did not enable the wildcard bar channels"
                        )
                    if any(
                        message.get("T") == "subscription"
                        and "*" in message["bars"]
                        and "*" in message["updatedBars"]
                        for message in messages
                    ):
                        break
            yield socket
    except (TimeoutError, WebSocketException) as error:
        raise ValueError(
            f"Alpaca {feed.upper()} stream interrupted: {type(error).__name__}"
        ) from None


async def record(
    client: Alpaca,
    directory: dict,
    window: Window,
    feed: str,
    out: Path,
) -> int:
    wanted = {
        row["symbol"]
        for row in directory_rows(directory, window.previous_days[-1], window.start)
        if not row["listing_reasons"]
    }
    if out.exists():
        raise ValueError(f"Receipt tape already exists: {out}")
    now = pd.Timestamp.now(tz="UTC")
    if window.start <= now:
        raise ValueError("Start recording before the selected feed's premarket begins")
    delay = (window.start - now).total_seconds() - 45
    if delay > 0:
        await asyncio.sleep(delay)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + ".partial")
    count = 0
    with partial.open("x", encoding="utf-8", buffering=1) as handle:
        async with subscribed_bars(client, feed) as socket:
            started = pd.Timestamp.now(tz="UTC")
            if started > window.start:
                raise ValueError(
                    "The subscription did not complete before the premarket window"
                )
            handle.write(
                json.dumps(
                    dict(
                        type="start",
                        schema=1,
                        date=window.date.isoformat(),
                        feed=feed,
                        started_at=started.isoformat(),
                        directory_sha256=directory["sha256"],
                    )
                )
                + "\n"
            )
            while True:
                seconds_left = (
                    window.cutoff - pd.Timestamp.now(tz="UTC")
                ).total_seconds()
                if seconds_left <= 0:
                    break
                try:
                    message = await asyncio.wait_for(socket.recv(), seconds_left)
                except TimeoutError:
                    break
                received = pd.Timestamp.now(tz="UTC")
                for event in json.loads(message):
                    if event.get("T") == "error":
                        raise ValueError(f"Alpaca stream error ({event['code']})")
                    if event.get("T") in {"b", "u"} and event.get("S") in wanted:
                        handle.write(
                            json.dumps(
                                dict(
                                    type="bar",
                                    received_at=received.isoformat(),
                                    event=event,
                                )
                            )
                            + "\n"
                        )
                        count += 1
            if not count:
                raise ValueError("The stream recorded no bars for the frozen directory")
            handle.write(
                json.dumps(
                    dict(
                        type="complete",
                        finished_at=pd.Timestamp.now(tz="UTC").isoformat(),
                        records=count,
                    )
                )
                + "\n"
            )
    partial.replace(out)
    return count


def read_recording(
    path: Path,
) -> tuple[dict, dict[str, list[tuple[pd.Timestamp, Bar]]]]:
    events: dict[str, list[tuple[pd.Timestamp, Bar]]] = {}
    with path.open(encoding="utf-8") as source:
        first = source.readline()
        header = json.loads(first) if first else {}
        if header.get("schema") != 1 or header.get("type") != "start":
            raise ValueError("Unsupported or missing receipt tape header")
        previous = timestamp(header["started_at"])
        count = 0
        for line in source:
            row = json.loads(line)
            if row.get("type") == "complete":
                if (
                    not count
                    or row["records"] != count
                    or timestamp(row["finished_at"]) < previous
                    or source.readline()
                ):
                    raise ValueError("Invalid receipt tape completion record")
                return {**header, "finished_at": row["finished_at"]}, events
            if row.get("type") != "bar" or row["event"].get("T") not in {"b", "u"}:
                raise ValueError("Unexpected event in receipt tape")
            observed = timestamp(row["received_at"])
            if observed < previous:
                raise ValueError("Receipt tape timestamps are out of order")
            previous = observed
            event = row["event"]
            events.setdefault(event["S"], []).append((observed, Bar.read(event)))
            count += 1
    raise ValueError("Receipt tape did not record successful completion")
