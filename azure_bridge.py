import asyncio
import base64
import json
import logging
import os
from contextlib import suppress
from urllib.parse import urlencode, urlsplit

import websockets
from fastapi import WebSocket, WebSocketDisconnect
from websockets.exceptions import InvalidStatus

from feedback import generate_feedback
from prompts import get_system_prompt
from scenarios import DEFAULT_SCENARIO, SCENARIOS

logger = logging.getLogger("voice")

# Stores the last completed call's transcript for on-demand feedback
_last_transcript: str = ""
_last_scenario_description: str = ""


def get_last_transcript():
    return _last_transcript, _last_scenario_description


def azure_connection_settings():
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    deployment = os.getenv("AZURE_OPENAI_REALTIME_DEPLOYMENT", "").strip()

    if not endpoint or not key or not deployment:
        raise ValueError("Set endpoint, API key and deployment name in .env.")

    parsed = urlsplit(endpoint)

    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("AZURE_OPENAI_ENDPOINT must start with https://")

    query = urlencode({"model": deployment})
    url = f"wss://{parsed.netloc}/openai/v1/realtime?{query}"

    return url, key


def session_configuration(scenario_key: str = DEFAULT_SCENARIO):
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": get_system_prompt(scenario_key),
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {"model": "whisper-1"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 600,
                        "create_response": True,
                        "interrupt_response": True,
                    },
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "voice": "alloy",
                },
            },
        },
    }


async def wait_for_event(azure, expected_type):
    async with asyncio.timeout(20):
        while True:
            event = json.loads(await azure.recv())

            if event.get("type") == "error":
                message = event.get("error", {}).get("message", "Azure configuration error")
                raise RuntimeError(message)

            if event.get("type") == expected_type:
                return event


async def voice_socket(browser: WebSocket, scenario: str = DEFAULT_SCENARIO):
    origin = browser.headers.get("origin")
    if origin and urlsplit(origin).netloc != browser.headers.get("host"):
        await browser.close(code=1008)
        return

    await browser.accept()
    tasks = []

    # Collect full conversation transcript for feedback
    conversation_lines = []
    scenario_description = SCENARIOS.get(scenario, {}).get("description", "")
    global _last_transcript, _last_scenario_description
    _last_scenario_description = scenario_description

    try:
        url, key = azure_connection_settings()
        logger.info("Connecting to Azure Realtime")

        async with websockets.connect(
            url,
            additional_headers={"api-key": key},
            open_timeout=20,
            max_size=16 * 1024 * 1024,
        ) as azure:
            created = await wait_for_event(azure, "session.created")
            logger.info("Azure session: %s", created["session"]["id"])

            await azure.send(json.dumps(session_configuration(scenario)))
            await wait_for_event(azure, "session.updated")

            await browser.send_json({"type": "ready"})
            logger.info("Azure audio input/output ready")

            audio_items = {}
            input_chunks = 0
            output_chunks = 0
            bot_turn_buffer = []

            async def browser_to_azure():
                nonlocal input_chunks

                while True:
                    message = await browser.receive()

                    if message["type"] == "websocket.disconnect":
                        return

                    audio = message.get("bytes")

                    if audio is not None:
                        if not audio:
                            continue

                        if len(audio) % 2 or len(audio) > 65536:
                            raise ValueError("Invalid PCM16 audio chunk")

                        input_chunks += 1

                        await azure.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(audio).decode("ascii"),
                        }))

                        if input_chunks % 100 == 0:
                            logger.info("Microphone chunks forwarded: %s", input_chunks)

                    elif message.get("text"):
                        control = json.loads(message["text"])

                        if control.get("type") == "truncate":
                            item_id = control.get("item_id")
                            item = audio_items.get(item_id)

                            if item is None:
                                continue

                            requested_ms = control.get("audio_end_ms")
                            if not isinstance(requested_ms, (int, float)):
                                continue

                            end_ms = max(0, min(int(requested_ms), item["samples"] // 24))

                            await azure.send(json.dumps({
                                "type": "conversation.item.truncate",
                                "item_id": item_id,
                                "content_index": item["content_index"],
                                "audio_end_ms": end_ms,
                            }))

            async def azure_to_browser():
                nonlocal output_chunks, bot_turn_buffer

                async for raw in azure:
                    event = json.loads(raw)
                    kind = event.get("type")

                    if kind == "input_audio_buffer.speech_started":
                        logger.info("User speaking")
                        await browser.send_json({"type": "user_started"})

                    elif kind == "input_audio_buffer.speech_stopped":
                        logger.info("User stopped")
                        await browser.send_json({"type": "user_stopped"})

                    elif kind == "conversation.item.input_audio_transcription.completed":
                        rep_text = event.get("transcript", "").strip()
                        if rep_text:
                            conversation_lines.append(f"REP: {rep_text}")

                    elif kind == "response.created":
                        bot_turn_buffer = []
                        await browser.send_json({
                            "type": "response_started",
                            "response_id": event["response"]["id"],
                        })

                    elif kind == "response.output_audio.delta":
                        encoded_audio = event.get("delta")
                        if not encoded_audio:
                            continue

                        item_id = event["item_id"]
                        decoded = base64.b64decode(encoded_audio)

                        item = audio_items.setdefault(item_id, {
                            "samples": 0,
                            "content_index": event.get("content_index", 0),
                        })
                        item["samples"] += len(decoded) // 2

                        output_chunks += 1

                        if output_chunks == 1 or output_chunks % 20 == 0:
                            logger.info(
                                "Bot audio → browser: chunk %s, %s bytes",
                                output_chunks, len(decoded),
                            )

                        await browser.send_json({
                            "type": "audio",
                            "audio": encoded_audio,
                            "item_id": item_id,
                            "response_id": event["response_id"],
                        })

                    elif kind == "response.output_audio_transcript.delta":
                        delta = event.get("delta", "")
                        bot_turn_buffer.append(delta)
                        await browser.send_json({
                            "type": "transcript",
                            "delta": delta,
                            "response_id": event["response_id"],
                        })

                    elif kind == "response.done":
                        response = event["response"]
                        status = response.get("status")

                        if bot_turn_buffer:
                            conversation_lines.append(f"BOT: {''.join(bot_turn_buffer)}")
                            bot_turn_buffer = []

                        logger.info("Response generation: %s", status)

                        await browser.send_json({
                            "type": "response_done",
                            "response_id": response["id"],
                            "status": status,
                        })

                        if status == "failed":
                            raise RuntimeError(
                                "Azure response failed: "
                                + json.dumps(response.get("status_details"))
                            )

                    elif kind == "error":
                        raise RuntimeError(
                            event.get("error", {}).get("message", "Azure Realtime error")
                        )

                raise RuntimeError("Azure closed the connection")

            tasks = [
                asyncio.create_task(browser_to_azure()),
                asyncio.create_task(azure_to_browser()),
            ]

            try:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    task.result()

            finally:
                for task in tasks:
                    task.cancel()

                await asyncio.gather(*tasks, return_exceptions=True)

    except WebSocketDisconnect:
        logger.info("Browser disconnected")

    except InvalidStatus as error:
        logger.error("Azure handshake rejected: HTTP %s", error.response.status_code)

        with suppress(Exception):
            await browser.send_json({
                "type": "error",
                "message": (
                    f"Azure rejected the connection: "
                    f"HTTP {error.response.status_code}. "
                    "Check the endpoint, key and deployment."
                ),
            })

    except Exception as error:
        logger.exception("Voice connection failed")

        with suppress(Exception):
            await browser.send_json({"type": "error", "message": str(error)})

    finally:
        for task in tasks:
            task.cancel()

        # Save transcript for on-demand feedback
        if len(conversation_lines) >= 2:
            _last_transcript = "\n".join(conversation_lines)

        with suppress(Exception):
            await browser.close()

        logger.info("Voice connection closed")
