"""Pygments lexer for SIP messages (RFC 3261)."""

from pygments import lexer, lexers, token


class SIPLexer(lexers.HttpLexer):
    """Lexer for Session Initiation Protocol (SIP) messages."""

    name = "Session Initiation Protocol"
    aliases = ["SIP", "sip"]

    tokens = {
        "root": [
            (
                r"(REGISTER|INVITE|ACK|CANCEL|BYE|OPTIONS|INFO|PRACK"
                r"|SUBSCRIBE|NOTIFY|REFER|UPDATE|MESSAGE|PUBLISH)"
                r"( +)([^ ]+)( +)(SIP)(/)(2\.0)(\r?\n|\Z)",
                lexer.bygroups(
                    token.Name.Function,
                    token.Text,
                    token.Name.Namespace,
                    token.Text,
                    token.Keyword.Reserved,
                    token.Operator,
                    token.Number,
                    token.Text,
                ),
                "headers",
            ),
            (
                r"(SIP)(/)(2\.0)( +)(\d{3})(?:( +)([^\r\n]*))?(\r?\n|\Z)",
                lexer.bygroups(
                    token.Keyword.Reserved,
                    token.Operator,
                    token.Number,
                    token.Text,
                    token.Number,
                    token.Text,
                    token.Name.Exception,
                    token.Text,
                ),
                "headers",
            ),
        ],
        "headers": [
            (
                r"([^\s:]+)( *)(:)( *)([^\r\n]+)(\r?\n|\Z)",
                lexers.HttpLexer.header_callback,
            ),
            (
                r"([\t ]+)([^\r\n]+)(\r?\n|\Z)",
                lexers.HttpLexer.continuous_header_callback,
            ),
            (r"\r?\n", token.Text, "content"),
        ],
        "content": [(r".+", lexers.HttpLexer.content_callback)],
    }
