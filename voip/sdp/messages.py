"""SDP message parsing and serialization as defined by RFC 4566."""

from __future__ import annotations

import dataclasses
from collections.abc import Generator

from .types import (
    Attribute,
    Bandwidth,
    ConnectionData,
    Field,
    IntField,
    MediaDescription,
    Origin,
    StrField,
    Timing,
)

__all__ = ["SessionDescription"]


# Ordered sequence of field descriptors; each carries letter, session_attr,
# is_list, media_attr and a parse classmethod/staticmethod.
FIELD_MAP: tuple[Field, ...] = (
    IntField(letter="v", session_attr="version"),
    Origin,
    StrField(letter="s", session_attr="name"),
    StrField(letter="i", session_attr="title", media_attr="title"),
    StrField(letter="u", session_attr="uri"),
    StrField(letter="e", session_attr="emails", is_list=True),
    StrField(letter="p", session_attr="phones", is_list=True),
    ConnectionData,
    Bandwidth,
    Timing,
    StrField(letter="r", session_attr="repeat"),
    StrField(letter="z", session_attr="zone"),
    Attribute,
    MediaDescription,
)

FIELD_BY_LETTER: dict[str, Field] = {field.letter: field for field in FIELD_MAP}


@dataclasses.dataclass
class SessionDescription:
    """Session Description Protocol message (RFC 4566).

    Holds all session-level and media-level fields in their canonical order.
    """

    version: int = 0
    origin: Origin | None = None
    name: str = "-"
    title: str | None = None
    uri: str | None = None
    emails: list[str] = dataclasses.field(default_factory=list)
    phones: list[str] = dataclasses.field(default_factory=list)
    connection: ConnectionData | None = None
    bandwidths: list[Bandwidth] = dataclasses.field(default_factory=list)
    timings: list[Timing] = dataclasses.field(default_factory=list)
    repeat: str | None = None
    zone: str | None = None
    attributes: list[Attribute] = dataclasses.field(default_factory=list)
    media: list[MediaDescription] = dataclasses.field(default_factory=list)

    @classmethod
    def parse(cls, data: bytes | str) -> SessionDescription:
        """Parse a SDP message from bytes or str."""
        text = data.decode() if isinstance(data, bytes) else data
        sdp = cls()
        current_media: MediaDescription | None = None
        for line in text.splitlines():
            current_media = sdp._apply_line(line.rstrip("\r"), current_media)
        return sdp

    def _apply_line(
        self, line: str, current_media: MediaDescription | None
    ) -> MediaDescription | None:
        """Apply a single SDP line to this session, return the active MediaDescription."""
        if not line or "=" not in line:
            return current_media
        letter, _, value = line.partition("=")
        if letter not in FIELD_BY_LETTER:
            return current_media
        field = FIELD_BY_LETTER[letter]
        parsed = field.parse(value)
        if letter == "m":
            self.media.append(parsed)
            return parsed
        if field.media_attr is not None and current_media is not None:
            return self._apply_to_media(
                current_media, field.media_attr, parsed, field.is_list
            )
        if field.is_list:
            getattr(self, field.session_attr).append(parsed)
        else:
            setattr(self, field.session_attr, parsed)
        return current_media

    @staticmethod
    def _apply_to_media(
        media: MediaDescription, attr: str, value: object, is_list: bool
    ) -> MediaDescription:
        """Apply a parsed field value to a MediaDescription, return it unchanged."""
        if is_list:
            getattr(media, attr).append(value)
        else:
            setattr(media, attr, value)
        return media

    def __bytes__(self) -> bytes:
        """Serialize to bytes."""
        return str(self).encode()

    def __str__(self) -> str:
        """Serialize to SDP text."""
        return "\r\n".join(self._lines()) + "\r\n"

    def _lines(self) -> Generator[str]:
        """Yield each SDP line in canonical field order."""
        for field in FIELD_MAP:
            if field.session_attr == "media":
                yield from (str(m) for m in self.media)
                continue
            value = getattr(self, field.session_attr)
            if field.is_list:
                yield from (f"{field.letter}={v}" for v in value)
            elif (
                value is not None
                and value != ""
                or field.session_attr in ("version", "name")
            ):
                yield f"{field.letter}={value}"
