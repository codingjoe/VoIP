"""Tests for the SDP Pygments lexer."""

import pytest

token = pytest.importorskip("pygments.token")
from voip.sdp.lexers import SDPLexer  # noqa: E402


class TestSDPLexer:
    def test_sdp_lexer__name(self):
        """Expose 'SDP' as the lexer name."""
        assert SDPLexer.name == "SDP"

    def test_sdp_lexer__aliases(self):
        """Expose 'sdp' as the lexer alias."""
        assert "sdp" in SDPLexer.aliases

    def test_sdp_lexer__version_field(self):
        """Tokenize the v= field."""
        lexer_instance = SDPLexer()
        tokens = list(lexer_instance.get_tokens("v=0\r\n"))
        token_types = [t for t, _ in tokens]
        assert token.Name.Tag in token_types
        assert token.Punctuation in token_types
        assert token.String in token_types

    def test_sdp_lexer__field_tag_value(self):
        """Include the field character in Name.Tag tokens."""
        lexer_instance = SDPLexer()
        tokens = list(lexer_instance.get_tokens("s=My Session\r\n"))
        tag_values = [v for t, v in tokens if t == token.Name.Tag]
        assert "s" in tag_values

    def test_sdp_lexer__attribute_flag(self):
        """Tokenize a flag attribute (no value) as Name.Attribute."""
        lexer_instance = SDPLexer()
        tokens = list(lexer_instance.get_tokens("a=recvonly\r\n"))
        attr_values = [v for t, v in tokens if t == token.Name.Attribute]
        assert "recvonly" in attr_values

    def test_sdp_lexer__attribute_with_value(self):
        """Tokenize an attribute with a value, splitting name and value."""
        lexer_instance = SDPLexer()
        tokens = list(lexer_instance.get_tokens("a=rtpmap:96 opus/48000/2\r\n"))
        token_types = [t for t, _ in tokens]
        assert token.Name.Attribute in token_types
        assert token.String in token_types

    def test_sdp_lexer__attribute_name_tagged(self):
        """Include the attribute name in Name.Attribute tokens."""
        lexer_instance = SDPLexer()
        tokens = list(lexer_instance.get_tokens("a=rtpmap:96 opus/48000/2\r\n"))
        attr_values = [v for t, v in tokens if t == token.Name.Attribute]
        assert any("rtpmap" in v for v in attr_values)

    def test_sdp_lexer__attribute_value_tagged(self):
        """Include the attribute value in String tokens for a= lines."""
        lexer_instance = SDPLexer()
        tokens = list(lexer_instance.get_tokens("a=rtpmap:96 opus/48000/2\r\n"))
        string_values = [v for t, v in tokens if t == token.String]
        assert any("96 opus/48000/2" in v for v in string_values)

    def test_sdp_lexer__multiple_fields(self):
        """Tokenize a multi-line SDP snippet."""
        lexer_instance = SDPLexer()
        data = "v=0\r\ns=Test\r\nt=0 0\r\n"
        tokens = list(lexer_instance.get_tokens(data))
        tag_values = [v for t, v in tokens if t == token.Name.Tag]
        assert "v" in tag_values
        assert "s" in tag_values
        assert "t" in tag_values

    def test_sdp_lexer__punctuation_equals(self):
        """Include the = separator as Punctuation."""
        lexer_instance = SDPLexer()
        tokens = list(lexer_instance.get_tokens("v=0\r\n"))
        punct_values = [v for t, v in tokens if t == token.Punctuation]
        assert "=" in punct_values

    def test_sdp_lexer__media_field(self):
        """Tokenize the m= field."""
        lexer_instance = SDPLexer()
        tokens = list(lexer_instance.get_tokens("m=audio 49170 RTP/AVP 0\r\n"))
        tag_values = [v for t, v in tokens if t == token.Name.Tag]
        assert "m" in tag_values

    def test_sdp_lexer__full_sdp(self):
        """Tokenize a full SDP without errors."""
        lexer_instance = SDPLexer()
        sdp = (
            "v=0\r\n"
            "o=alice 2890844526 2890844526 IN IP4 pc33.atlanta.com\r\n"
            "s=Session SDP\r\n"
            "c=IN IP4 224.2.1.1/127/3\r\n"
            "t=0 0\r\n"
            "m=audio 49170 RTP/AVP 0\r\n"
            "a=rtpmap:0 PCMU/8000\r\n"
        )
        tokens = list(lexer_instance.get_tokens(sdp))
        assert len(tokens) > 0
