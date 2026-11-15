"""Resolve peer photos from every source Telegram offers them through."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from telethon.tl import functions, types
from telethon.tl.types import Channel, Chat, User

AVATAR_SOURCE = "avatars"
MESSAGE_SOURCE = "messages"
PHOTO_SOURCES = (AVATAR_SOURCE, MESSAGE_SOURCE)
THUMBNAIL_TARGET_PIXELS = 320

# On this account/network, a photo download can sit unanswered for minutes
# at a time — not because Telethon or the network is broken (a fresh,
# isolated connection downloads every size of the same photo in well under
# a second), but because the single shared MTProto connection this server
# keeps open all day is sometimes busy churning through a large backlog of
# other updates first, and our GetFileRequest just waits in line behind it.
# Bound every download attempt so callers get a real answer instead of
# hanging forever. The first (full-resolution) attempt gets a generous
# budget since it usually succeeds almost instantly and only occasionally
# hits the slow path; the fallback attempt is capped much shorter so the
# *total* worst case stays bounded instead of multiplying a long timeout
# across every available size.
PHOTO_DOWNLOAD_TIMEOUT_SECONDS = float(
    os.getenv("TELEGRAM_PHOTO_DOWNLOAD_TIMEOUT_SECONDS", "30")
)
PHOTO_DOWNLOAD_FALLBACK_TIMEOUT_SECONDS = float(
    os.getenv("TELEGRAM_PHOTO_DOWNLOAD_FALLBACK_TIMEOUT_SECONDS", "10")
)


class PhotoDownloadTimedOut(Exception):
    """Raised when every size of a photo timed out while downloading."""


class UnknownPhotoSource(ValueError):
    """Raised when a caller asks for a source that does not exist."""


@dataclass(frozen=True)
class PhotoReference:
    """One retrievable photo plus the identifier used to open it again."""

    identifier: int
    photo: Any
    is_current: bool
    taken_at: Optional[Any] = None
    caption: str = ""

    def describe(self) -> dict:
        described = {
            "id": self.identifier,
            "date": self.taken_at,
            "is_current": self.is_current,
        }
        if self.caption:
            described["caption"] = self.caption
        return described


def validate_source(source: str) -> str:
    normalised_source = (source or AVATAR_SOURCE).strip().lower()
    if normalised_source not in PHOTO_SOURCES:
        raise UnknownPhotoSource(
            f"Unknown photo source '{source}'. Expected one of: {', '.join(PHOTO_SOURCES)}."
        )
    return normalised_source


def _current_photo_id(entity: Any) -> Optional[int]:
    entity_photo = getattr(entity, "photo", None)
    return getattr(entity_photo, "photo_id", None)


def _peer_supports_native_avatar_history(entity: Any) -> bool:
    return isinstance(entity, User)


async def _list_user_avatar_references(client, entity, limit: int) -> List[PhotoReference]:
    retrieved = await client(
        functions.photos.GetUserPhotosRequest(user_id=entity, offset=0, max_id=0, limit=limit)
    )
    current_photo_id = _current_photo_id(entity)
    return [
        PhotoReference(
            identifier=photo.id,
            photo=photo,
            is_current=photo.id == current_photo_id,
            taken_at=getattr(photo, "date", None),
        )
        for photo in retrieved.photos
    ]


async def _list_chat_avatar_references(client, entity, limit: int) -> List[PhotoReference]:
    service_messages = await client.get_messages(
        entity, limit=limit, filter=types.InputMessagesFilterChatPhotos()
    )
    current_photo_id = _current_photo_id(entity)

    references: List[PhotoReference] = []
    for message in service_messages:
        changed_photo = getattr(getattr(message, "action", None), "photo", None)
        if changed_photo is None:
            continue
        references.append(
            PhotoReference(
                identifier=changed_photo.id,
                photo=changed_photo,
                is_current=changed_photo.id == current_photo_id,
                taken_at=getattr(message, "date", None),
            )
        )
    return references


async def _list_message_photo_references(client, entity, limit: int) -> List[PhotoReference]:
    photo_messages = await client.get_messages(
        entity, limit=limit, filter=types.InputMessagesFilterPhotos()
    )
    return [
        PhotoReference(
            identifier=message.id,
            photo=message.photo,
            is_current=False,
            taken_at=getattr(message, "date", None),
            caption=getattr(message, "message", "") or "",
        )
        for message in photo_messages
        if getattr(message, "photo", None) is not None
    ]


async def list_photo_references(client, entity, source: str, limit: int) -> List[PhotoReference]:
    """List the newest ``limit`` photos of ``entity`` from the requested source."""
    resolved_source = validate_source(source)

    if resolved_source == MESSAGE_SOURCE:
        return await _list_message_photo_references(client, entity, limit)
    if _peer_supports_native_avatar_history(entity):
        return await _list_user_avatar_references(client, entity, limit)
    return await _list_chat_avatar_references(client, entity, limit)


async def find_photo_reference(
    client, entity, source: str, identifier: Optional[int], search_depth: int
) -> Optional[PhotoReference]:
    """Fetch a message photo directly, or search avatar history by photo id."""
    resolved_source = validate_source(source)
    if resolved_source == MESSAGE_SOURCE and identifier is not None:
        message = await client.get_messages(entity, ids=identifier)
        # `Message.photo` also exposes web-page preview images, but this id
        # space is defined by InputMessagesFilterPhotos, which only matches
        # attached photos (same convention as get_media_label).
        if not isinstance(getattr(message, "media", None), types.MessageMediaPhoto):
            return None
        photo = getattr(message, "photo", None)
        if photo is None:
            return None
        return PhotoReference(
            identifier=message.id,
            photo=photo,
            is_current=False,
            taken_at=getattr(message, "date", None),
            caption=getattr(message, "message", "") or "",
        )

    references = await list_photo_references(client, entity, resolved_source, search_depth)
    if not references:
        return None
    if identifier is None:
        current_avatar = next((each for each in references if each.is_current), None)
        return current_avatar or references[0]
    return next((each for each in references if each.identifier == identifier), None)


def _thumbnail_size_index(photo: Any) -> Optional[int]:
    available_sizes = getattr(photo, "sizes", None) or []
    widths = [(index, getattr(size, "w", 0) or 0) for index, size in enumerate(available_sizes)]
    within_target = [index for index, width in widths if 0 < width <= THUMBNAIL_TARGET_PIXELS]
    if within_target:
        return within_target[-1]
    return None


def _smallest_size_index(photo: Any) -> Optional[int]:
    """The cheapest real size to fall back to — smallest width, excluding
    zero-width/undeclared sizes (e.g. ``PhotoPathSize``) which are not a
    plain downloadable image."""
    available_sizes = getattr(photo, "sizes", None) or []
    candidates = [
        (index, getattr(size, "w", 0) or 0) for index, size in enumerate(available_sizes)
    ]
    candidates = [item for item in candidates if item[1] > 0]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[1])[0]


async def _download_one_size(
    client, photo: Any, size_index: Optional[int], timeout: float
) -> Optional[bytes]:
    kwargs = {} if size_index is None else {"thumb": size_index}
    try:
        return await asyncio.wait_for(
            client.download_media(photo, file=bytes, **kwargs),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return None


async def download_photo_bytes(
    client, reference: PhotoReference, thumbnail: bool = False
) -> bytes:
    """Download one referenced photo straight to memory, never to disk.

    On this account/network, a photo download can occasionally sit
    unanswered for minutes — most likely the shared connection catching up
    on a backlog of other updates first, not a broken download. Every
    attempt here is time-boxed so callers always get either real bytes or a
    prompt ``PhotoDownloadTimedOut`` — never an indefinite hang. The primary
    attempt gets a generous budget (it usually finishes in well under a
    second); if it times out, exactly one cheap fallback to the smallest
    available size is tried with a short budget, keeping the total worst
    case bounded instead of stacking a long timeout across every size.
    """
    primary_index = _thumbnail_size_index(reference.photo) if thumbnail else None
    downloaded = await _download_one_size(
        client, reference.photo, primary_index, PHOTO_DOWNLOAD_TIMEOUT_SECONDS
    )
    if downloaded:
        return downloaded

    fallback_index = _smallest_size_index(reference.photo)
    if fallback_index is not None and fallback_index != primary_index:
        downloaded = await _download_one_size(
            client, reference.photo, fallback_index, PHOTO_DOWNLOAD_FALLBACK_TIMEOUT_SECONDS
        )
        if downloaded:
            return downloaded

    raise PhotoDownloadTimedOut(
        f"Photo {reference.identifier} timed out after "
        f"{PHOTO_DOWNLOAD_TIMEOUT_SECONDS:.0f}s (and a "
        f"{PHOTO_DOWNLOAD_FALLBACK_TIMEOUT_SECONDS:.0f}s fallback attempt). "
        "Telegram's response is likely just delayed behind other traffic on "
        "the shared connection right now — retry in a bit."
    )


def peer_supports_source(entity: Any, source: str) -> bool:
    """Whether the peer can serve the source at all, used to explain empty results."""
    if validate_source(source) == MESSAGE_SOURCE:
        return isinstance(entity, (Chat, Channel, User))
    return True


# --- shared-client download serialisation and reconnect (ported 15/11/2026) ---
# Telethon #809: concurrent GetFile calls on one client race. Telethon #1536: a
# half-alive MTProto sender hangs GetFile forever. Downloads are therefore
# serialised per client and a hung attempt is followed by one forced reconnect.
_download_locks: Dict[int, asyncio.Lock] = {}


def download_lock_for(client) -> asyncio.Lock:
    """The asyncio.Lock that serialises downloads on this client."""
    key = id(client)
    lock = _download_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _download_locks[key] = lock
    return lock


async def force_reconnect_client(client) -> None:
    """Reconnect the shared Telethon client after a hung GetFile."""
    from telegram_mcp.runtime import _force_reconnect

    await _force_reconnect(client)
