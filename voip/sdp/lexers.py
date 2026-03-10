"""Pygments lexer for SDP messages (RFC 4566)."""

import re
from collections.abc import Generator

from pygments import lexer, token

__all__ = ["SDPLexer"]


class SDPLexer(lexer.RegexLexer):
    """Lexer for Session Description Protocol (SDP) messages (RFC 4566)."""

    name = "SDP"
    aliases = ["sdp"]
    filenames = ["*.sdp"]
    mimetypes = ["application/sdp"]

    def attribute_callback(
        self,
        match: re.Match,
    ) -> Generator:
        """Yield tokens for an a= attribute line."""
        yield match.start(1), token.Name.Tag, match.group(1)
        yield match.start(2), token.Punctuation, match.group(2)
        attr_val = match.group(3)
        colon_pos = attr_val.find(":")
        if colon_pos >= 0:
            yield match.start(3), token.Name.Attribute, attr_val[: colon_pos + 1]
            yield (
                match.start(3) + colon_pos + 1,
                token.String,
                attr_val[colon_pos + 1 :],
            )
        else:
            yield match.start(3), token.Name.Attribute, attr_val
        yield match.start(4), token.Text, match.group(4)

    tokens = {
        "root": [
            (
                r"(a)(=)([^\r\n]*)(\r?\n|\Z)",
                attribute_callback,
            ),
            (
                r"([vocsitbzruepmc])(=)([^\r\n]*)(\r?\n|\Z)",
                lexer.bygroups(
                    token.Name.Tag,
                    token.Punctuation,
                    token.String,
                    token.Text,
                ),
            ),
        ],
    }
