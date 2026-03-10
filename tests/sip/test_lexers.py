"""Tests for the SIP Pygments lexer."""

import pytest

token = pytest.importorskip("pygments.token")
from voip.sip.lexers import SIPLexer  # noqa: E402


class TestSIPLexer:
    def test_sip_lexer__name(self):
        """Expose 'SIP' as the lexer name."""
        assert SIPLexer.name == "Session Initiation Protocol"

    def test_sip_lexer__aliases(self):
        """Expose 'sip' as the lexer alias."""
        assert "sip" in SIPLexer.aliases

    @pytest.mark.parametrize(
        "method",
        [
            "REGISTER",
            "INVITE",
            "ACK",
            "CANCEL",
            "BYE",
            "OPTIONS",
            "INFO",
            "PRACK",
            "SUBSCRIBE",
            "NOTIFY",
            "REFER",
            "UPDATE",
            "MESSAGE",
            "PUBLISH",
        ],
    )
    def test_sip_lexer__request(self, method):
        """Tokenize a SIP request first line."""
        lexer = SIPLexer()
        data = f"{method} sip:bob@biloxi.com SIP/2.0\r\n\r\n"
        tokens = list(lexer.get_tokens(data))
        token_types = [t for t, _ in tokens]
        assert token.Name.Function in token_types
        assert token.Name.Namespace in token_types
        assert token.Number in token_types

    def test_sip_lexer__response(self):
        """Tokenize a SIP response first line."""
        lexer = SIPLexer()
        data = "SIP/2.0 200 OK\r\n\r\n"
        tokens = list(lexer.get_tokens(data))
        token_types = [t for t, _ in tokens]
        assert token.Number in token_types
        assert token.Operator in token_types

    def test_sip_lexer__headers(self):
        """Tokenize SIP headers."""
        lexer = SIPLexer()
        data = "SIP/2.0 200 OK\r\nVia: SIP/2.0/UDP pc33.atlanta.com\r\n\r\n"
        tokens = list(lexer.get_tokens(data))
        values = [v for _, v in tokens]
        assert "Via" in values

    def test_sip_lexer__request_with_body(self):
        """Tokenize a SIP request with a body."""
        lexer = SIPLexer()
        data = "INVITE sip:bob@biloxi.com SIP/2.0\r\nContent-Length: 4\r\n\r\ntest"
        tokens = list(lexer.get_tokens(data))
        token_types = [t for t, _ in tokens]
        assert token.Name.Function in token_types

    def test_sip_lexer__sdp_body(self):
        """Tokenize a SIP request with an SDP body using the SDPLexer."""
        lexer = SIPLexer()
        data = "INVITE sip:bob@biloxi.com SIP/2.0\r\nContent-Length: 13\r\n\r\nv=0\r\na=recvonly\r\n"
        tokens = list(lexer.get_tokens(data))
        token_types = [t for t, _ in tokens]
        assert token.Name.Tag in token_types
        assert token.Name.Attribute in token_types

    def test_sip_lexer__text_token(self):
        """Include Text tokens in tokenized output."""
        lexer = SIPLexer()
        data = "INVITE sip:bob@biloxi.com SIP/2.0\r\n\r\n"
        tokens = list(lexer.get_tokens(data))
        assert any(t == token.Text for t, _ in tokens)
