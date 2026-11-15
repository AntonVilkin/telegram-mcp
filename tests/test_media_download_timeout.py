"""The bounded non-photo download: a hung GetFile must end in a clean message.

Ported 15/11/2026 with the sync to upstream; guards the timeout + single
reconnect retry so a stuck media connection cannot hang a tool call forever.
"""

import asyncio

import pytest

from telegram_mcp.tools import media


class HangingClient:
    """download_media that never answers, like a half-alive MTProto sender."""

    def __init__(self):
        self.calls = 0

    async def download_media(self, message, *, file, progress_callback=None):
        self.calls += 1
        await asyncio.sleep(30)
        return file


def test_hung_download_times_out_after_one_reconnect(monkeypatch):
    client = HangingClient()
    reconnects = []

    async def fake_reconnect(cl):
        reconnects.append(cl)

    monkeypatch.setattr(media, "force_reconnect_client", fake_reconnect)
    monkeypatch.setattr(media, "MEDIA_DOWNLOAD_TIMEOUT_SECONDS", 0.01)

    with pytest.raises(media._MediaDownloadTimedOut):
        asyncio.run(
            media._bounded_client_download_media(client, object(), file="/tmp/never-written")
        )

    assert client.calls == 2, "expected the original attempt plus one retry"
    assert len(reconnects) == 1, "expected exactly one forced reconnect between attempts"


def test_successful_download_returns_the_path(monkeypatch):
    class WorkingClient:
        async def download_media(self, message, *, file, progress_callback=None):
            return file

    monkeypatch.setattr(media, "MEDIA_DOWNLOAD_TIMEOUT_SECONDS", 5)

    result = asyncio.run(
        media._bounded_client_download_media(WorkingClient(), object(), file="/tmp/ok")
    )

    assert result == "/tmp/ok"
