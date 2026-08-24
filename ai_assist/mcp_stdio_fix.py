"""
Patched MCP stdio client that fixes the buffering issue with FastMCP servers.

The issue: MCP SDK uses buffer size 0 for memory streams, which can cause
race conditions where the stdout_reader task starts but hasn't set up its
async iteration before the server sends its first response.

Solution: Use a small buffer (10) to allow messages to queue.
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import TextIO

import anyio
import anyio.lowlevel
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from anyio.streams.text import TextReceiveStream
from mcp import types
from mcp.client import stdio as mcp_stdio
from mcp.shared.message import SessionMessage

# Import the helper functions from the original module
_get_executable_command = mcp_stdio._get_executable_command
_create_platform_compatible_process = mcp_stdio._create_platform_compatible_process
get_default_environment = mcp_stdio.get_default_environment
StdioServerParameters = mcp_stdio.StdioServerParameters

logger = logging.getLogger(__name__)


@asynccontextmanager
async def stdio_client_fixed(server: StdioServerParameters, errlog: TextIO = sys.stderr):
    """
    Fixed stdio client with proper buffering.

    Changes from original:
    - Use buffer size 10 instead of 0 for memory streams
    - Add better error logging
    """
    read_stream: MemoryObjectReceiveStream[SessionMessage | Exception]
    read_stream_writer: MemoryObjectSendStream[SessionMessage | Exception]

    write_stream: MemoryObjectSendStream[SessionMessage]
    write_stream_reader: MemoryObjectReceiveStream[SessionMessage]

    # FIXED: Use buffer size 10 instead of 0
    read_stream_writer, read_stream = anyio.create_memory_object_stream(10)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(10)

    try:
        command = _get_executable_command(server.command)

        process = await _create_platform_compatible_process(
            command=command,
            args=server.args,
            env=({**get_default_environment(), **server.env} if server.env is not None else get_default_environment()),
            errlog=errlog,
            cwd=server.cwd,
        )
    except OSError:
        await read_stream.aclose()
        await write_stream.aclose()
        await read_stream_writer.aclose()
        await write_stream_reader.aclose()
        raise

    async def stdout_reader():
        assert process.stdout, "Opened process is missing stdout"

        logger.debug("[FIX] stdout_reader starting...")
        try:
            async with read_stream_writer:
                buffer = ""
                logger.debug("[FIX] Creating TextReceiveStream...")
                async for chunk in TextReceiveStream(
                    process.stdout,
                    encoding=server.encoding,
                    errors=server.encoding_error_handler,
                ):
                    logger.debug(f"[FIX] Received chunk: {repr(chunk[:100])}")
                    lines = (buffer + chunk).split("\n")
                    buffer = lines.pop()

                    for line in lines:
                        if not line.strip():  # Skip empty lines
                            continue

                        logger.debug(f"[FIX] Processing line: {line[:100]}")
                        try:
                            message = types.jsonrpc_message_adapter.validate_json(line)
                            session_message = SessionMessage(message)
                        except Exception as exc:
                            logger.error(f"Failed to parse JSONRPC message: {line[:100]}")
                            logger.exception("Parse error")
                            try:
                                await read_stream_writer.send(exc)
                            except anyio.ClosedResourceError, anyio.BrokenResourceError:
                                return
                        else:
                            await read_stream_writer.send(session_message)
                logger.debug("[FIX] TextReceiveStream ended (EOF)")
        except (anyio.ClosedResourceError, anyio.BrokenResourceError) as e:
            logger.info(f"[FIX] Stream closed in stdout_reader: {e}")
            await anyio.lowlevel.checkpoint()
        except Exception as e:
            logger.error(f"[FIX] stdout_reader error: {e}")
            raise

    async def stdin_writer():
        assert process.stdin, "Opened process is missing stdin"

        try:
            async with write_stream_reader:
                async for session_message in write_stream_reader:
                    json = session_message.message.model_dump_json(by_alias=True, exclude_none=True)
                    logger.debug(f"[FIX] stdin_writer sending: {json[:200]}")
                    await process.stdin.send(
                        (json + "\n").encode(
                            encoding=server.encoding,
                            errors=server.encoding_error_handler,
                        )
                    )
                    logger.debug("[FIX] stdin_writer sent message")
        except anyio.ClosedResourceError, anyio.BrokenResourceError:
            await anyio.lowlevel.checkpoint()
        except Exception as e:
            logger.error(f"stdin_writer error: {e}")
            raise

    # On POSIX `process` is an anyio ``Process`` (an async context manager); the
    # Windows-only ``FallbackProcess`` returned on that platform is not.
    async with (
        anyio.create_task_group() as tg,
        process,  # type: ignore[union-attr]
    ):
        tg.start_soon(stdout_reader)
        tg.start_soon(stdin_writer)

        # Give tasks a chance to start before yielding
        logger.debug("[FIX] Giving tasks a moment to start...")
        await anyio.sleep(0.1)
        logger.debug("[FIX] Yielding streams")

        try:
            yield read_stream, write_stream
        finally:
            if process.stdin:
                try:
                    await process.stdin.aclose()
                except Exception:
                    pass

            # Wait for process to exit
            try:
                with anyio.fail_after(2.0):
                    await process.wait()
            except TimeoutError:
                process.terminate()
                try:
                    with anyio.fail_after(1.0):
                        await process.wait()
                except TimeoutError:
                    process.kill()
