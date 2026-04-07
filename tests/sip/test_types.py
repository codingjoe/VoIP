import ipaddress

import pytest
from voip.sip import SipURI
from voip.sip.messages import Response
from voip.sip.types import CallerID


class TestSipURI:
    @pytest.mark.parametrize(
        "uri_str, expected_uri_obj",
        [
            # domain
            (
                "sip:alice@example.com",
                SipURI(scheme="sip", user="alice", host="example.com", port=5060),
            ),
            (
                "sips:alice@example.com",
                SipURI(scheme="sips", user="alice", host="example.com", port=5061),
            ),
            (
                "sip:alice@example.com:4050",
                SipURI(scheme="sip", user="alice", host="example.com", port=4050),
            ),
            (
                "sips:alice@example.com:4051",
                SipURI(scheme="sips", user="alice", host="example.com", port=4051),
            ),
            # ipv4
            (
                "sip:alice@192.168.1.1",
                SipURI(scheme="sip", user="alice", host="192.168.1.1", port=5060),
            ),
            (
                "sips:alice@192.168.1.1",
                SipURI(scheme="sips", user="alice", host="192.168.1.1", port=5061),
            ),
            (
                "sip:alice@192.168.1.1:4050",
                SipURI(scheme="sip", user="alice", host="192.168.1.1", port=4050),
            ),
            (
                "sips:alice@192.168.1.1:4051",
                SipURI(scheme="sips", user="alice", host="192.168.1.1", port=4051),
            ),
            # ipv6
            (
                "sip:alice@[::1]",
                SipURI(scheme="sip", user="alice", host="::1", port=5060),
            ),
            (
                "sips:alice@[::1]",
                SipURI(scheme="sips", user="alice", host="::1", port=5061),
            ),
            (
                "sip:alice@[::1]:4050",
                SipURI(scheme="sip", user="alice", host="::1", port=4050),
            ),
            (
                "sips:alice@[::1]:4051",
                SipURI(scheme="sips", user="alice", host="::1", port=4051),
            ),
            # uri-parameters
            (
                "sip:alice@example.com;transport=tcp",
                SipURI(
                    scheme="sip",
                    user="alice",
                    host="example.com",
                    port=5060,
                    parameters={"transport": "tcp"},
                ),
            ),
            (
                "sip:alice@example.com;transport=udp;ttl=15",
                SipURI(
                    scheme="sip",
                    user="alice",
                    host="example.com",
                    port=5060,
                    parameters={"transport": "udp", "ttl": "15"},
                ),
            ),
            # headers
            (
                "sip:alice@example.com?foo=bar",
                SipURI(
                    scheme="sip",
                    user="alice",
                    host="example.com",
                    port=5060,
                    headers={"foo": "bar"},
                ),
            ),
            (
                "sip:alice@example.com?tag=12345&foo=bar",
                SipURI(
                    scheme="sip",
                    user="alice",
                    host="example.com",
                    port=5060,
                    headers={"tag": "12345", "foo": "bar"},
                ),
            ),
            (
                r"sip:%61lice@atlanta.com;transport=TCP",
                SipURI(
                    scheme="sip",
                    user="alice",
                    host="atlanta.com",
                    port=5060,
                    parameters={"transport": "TCP"},
                ),
            ),
            (
                r"sip:atlanta.com;method=REGISTER?to=alice%40atlanta.com",
                SipURI(
                    scheme="sip",
                    user=None,
                    host="atlanta.com",
                    port=5060,
                    parameters={"method": "REGISTER"},
                    headers={"to": "alice@atlanta.com"},
                ),
            ),
        ],
    )
    def test_parse_valid(self, uri_str, expected_uri_obj):
        """Parse scheme, user, host and optional port from a valid SIP URI."""
        assert SipURI.parse(uri_str) == expected_uri_obj

    @pytest.mark.parametrize(
        "uri_str",
        [
            "http://example.com",  # wrong scheme
            "sip:@example.com",  # missing user
            "sip:@example.com:invalid-port",  # non-integer port
        ],
    )
    def test_parse_invalid(self, uri_str):
        """Raise ValueError when parsing an invalid SIP URI."""
        with pytest.raises(ValueError):
            SipURI.parse(uri_str)

    @pytest.mark.parametrize(
        "uri_obj, expected_uri_str",
        [
            (
                SipURI(scheme="sip", user="alice", host="example.com", port=5061),
                "sip:alice@example.com:5061",
            ),
            (
                SipURI(
                    scheme="sip",
                    user="alice",
                    host="example.com",
                    port=5060,
                    parameters={"transport": "TCP"},
                ),
                "sip:alice@example.com:5060;transport=TCP",
            ),
            (
                SipURI(
                    scheme="sip",
                    user="alice",
                    host="example.com",
                    port=5060,
                    headers={"foo": "bar"},
                ),
                "sip:alice@example.com:5060?foo=bar",
            ),
            (
                SipURI(
                    scheme="sip",
                    user="alice",
                    host="example.com",
                    port=5060,
                    parameters={"transport": "TCP"},
                    headers={"foo": "bar"},
                ),
                "sip:alice@example.com:5060;transport=TCP?foo=bar",
            ),
            # IPv6
            (
                SipURI(scheme="sip", user="alice", host="::1", port=5060),
                "sip:alice@[::1]:5060",
            ),
            (
                SipURI(
                    scheme="sip",
                    user="alice",
                    host=ipaddress.IPv6Address("::1"),
                    port=5060,
                ),
                "sip:alice@[::1]:5060",
            ),
            # IPv4
            (
                SipURI(scheme="sip", user="alice", host="127.0.0.1", port=5060),
                "sip:alice@127.0.0.1:5060",
            ),
            (
                SipURI(
                    scheme="sip",
                    user="alice",
                    host=ipaddress.IPv4Address("127.0.0.1"),
                    port=5060,
                ),
                "sip:alice@127.0.0.1:5060",
            ),
            # password in user-info
            (
                SipURI(
                    scheme="sip",
                    user="alice",
                    password="secret",  # noqa: S106
                    host="example.com",
                    port=5060,
                ),
                "sip:alice:secret@example.com:5060",
            ),
            # flag URI parameter (value=None) in __str__
            (
                SipURI(
                    scheme="sip",
                    user="alice",
                    host="example.com",
                    port=5060,
                    parameters={"lr": None},
                ),
                "sip:alice@example.com:5060;lr",
            ),
        ],
    )
    def test_str(self, uri_obj, expected_uri_str):
        """Test string representation of SipUri objects."""
        assert str(uri_obj) == expected_uri_str

    @pytest.mark.parametrize(
        "uri_str, expected_uri_obj",
        [
            # flag URI parameter (;lr with no value)
            (
                "sip:alice@example.com;lr",
                SipURI(
                    scheme="sip",
                    user="alice",
                    host="example.com",
                    port=5060,
                    parameters={"lr": None},
                ),
            ),
            # header without '=' value
            (
                "sip:alice@example.com?Subject",
                SipURI(
                    scheme="sip",
                    user="alice",
                    host="example.com",
                    port=5060,
                    headers={"Subject": ""},
                ),
            ),
        ],
    )
    def test_parse__flag_parameter_and_valueless_header(
        self, uri_str, expected_uri_obj
    ):
        """Parse flag URI parameters and valueless headers."""
        assert SipURI.parse(uri_str) == expected_uri_obj

    def test_maddr__with_parameter(self):
        """Parse NetworkAddress from maddr URI parameter."""
        uri = SipURI.parse("sip:alice@example.com;maddr=192.0.2.1:5060")
        assert uri.maddr == (ipaddress.IPv4Address("192.0.2.1"), 5060)

    def test_maddr__without_parameter(self):
        """Fall back to host:port when maddr parameter is absent."""
        uri = SipURI.parse("sip:alice@192.0.2.2:5060")
        result = uri.maddr
        assert result.port == 5060

    def test_ttl__returns_value(self):
        """Return the ttl URI parameter value as a string."""
        uri = SipURI.parse("sip:alice@example.com;ttl=30")
        assert uri.ttl == 30

    def test_ttl__absent(self):
        """Return None when the ttl parameter is absent."""
        assert SipURI.parse("sip:alice@example.com").ttl is None

    def test_transport__sips_returns_tls(self):
        """Return 'TLS' for sips: URIs that have no transport parameter."""
        uri = SipURI.parse("sips:alice@example.com")
        assert uri.transport == "TLS"

    def test_transport__sip_without_parameter_returns_none(self):
        """Return None for a plain sip: URI without transport parameter."""
        uri = SipURI.parse("sip:alice@example.com")
        assert uri.transport == "TLS"

    def test_transport__explicit_parameter(self):
        """Return explicit transport parameter value."""
        uri = SipURI.parse("sip:alice@example.com;transport=udp")
        assert uri.transport == "UDP"

    def test_isinstance__str(self):
        """SipUri instances are also plain str instances."""
        assert isinstance(SipURI.parse("sip:alice@example.com"), str)


def _ok() -> Response:
    return Response(status_code=200, phrase="OK")


def _trying() -> Response:
    return Response(status_code=100, phrase="Trying")


class TestCallerID:
    def test_display_name__quoted(self):
        """Parse a quoted display name before the angle bracket."""
        assert (
            CallerID('"Alice Smith" <sip:alice@example.com>').display_name
            == "Alice Smith"
        )

    def test_display_name__unquoted(self):
        """Parse an unquoted display name before the angle bracket."""
        assert CallerID("Alice <sip:alice@example.com>").display_name == "Alice"

    def test_display_name__absent(self):
        """Return None when there is no display name."""
        assert CallerID("sip:alice@example.com").display_name is None

    def test_user__present(self):
        """Extract the SIP user part."""
        assert CallerID("sip:08001234567@example.com").user == "08001234567"

    def test_user__absent(self):
        """Return None when no SIP user is present."""
        assert CallerID("example.com").user is None

    def test_host__present(self):
        """Extract the carrier domain from the SIP URI."""
        assert CallerID("sip:alice@example.com").host == "example.com"

    def test_host__absent(self):
        """Return None when no host is found."""
        assert CallerID("plain string").host is None

    def test_tag__present(self):
        """Extract the dialog tag parameter."""
        assert CallerID("sip:alice@example.com;tag=abc123").tag == "abc123"

    def test_tag__absent(self):
        """Return None when no tag parameter is present."""
        assert CallerID("sip:alice@example.com").tag is None

    def test_repr__long_user(self):
        """Mask all but the last four chars of a long caller string."""
        assert (
            repr(CallerID('"08001234567" <sip:08001234567@telefonica.de>;tag=abc'))
            == "*******4567@telefonica.de"
        )

    def test_repr__short_user(self):
        """Show all characters when the name is four characters or fewer."""
        assert repr(CallerID("sip:alice@example.com")) == "*lice@example.com"

    def test_repr__no_user_no_host(self):
        """Fall back to asterisks when neither user nor host can be parsed."""
        assert "****" in repr(CallerID(""))

    def test_repr__no_host(self):
        """Show only masked user when there is no carrier domain."""
        masked = repr(CallerID("notasipuri"))
        assert "@" not in masked

    def test_uri__sip(self):
        """Extract a SipUri from a SIP CallerID."""
        assert isinstance(CallerID("sip:alice@example.com").uri, SipURI)

    def test_uri__sip_angle_brackets(self):
        """Extract SipUri from a CallerID with angle-bracket notation."""
        assert isinstance(
            CallerID('"Alice" <sip:alice@example.com>;tag=abc').uri, SipURI
        )

    def test_uri__absent(self):
        """Return None when no URI is present."""
        assert CallerID("plain string").uri is None

    def test_uri__unparseable(self):
        """Return None when the URI-like string is not valid for any parser."""
        with pytest.raises(ValueError):
            assert CallerID("sip:@invalid").uri

    def test_host__tel_absent(self):
        """Return None for host when the CallerID is a tel URI."""
        assert CallerID("tel:+15551234567").host is None


class TestMaskCaller:
    def test_mask_caller__with_display_name(self):
        """Mask all but last four chars of a quoted display name."""
        from voip.sip.types import _mask_caller

        assert (
            _mask_caller('"08001234567" <sip:08001234567@example.com>;tag=abc')
            == "*******4567"
        )

    def test_mask_caller__bare_uri(self):
        """Mask user part from a bare SIP URI."""
        from voip.sip.types import _mask_caller

        assert _mask_caller("sip:alice@example.com") == "*lice"

    def test_mask_caller__short_name(self):
        """Return the name unchanged when it is four characters or fewer."""
        from voip.sip.types import _mask_caller

        assert _mask_caller("sip:bob@example.com") == "bob"

    def test_mask_caller__long_name(self):
        """Mask all but last four characters of a long username."""
        from voip.sip.types import _mask_caller

        result = _mask_caller("sip:verylonguser@example.com")
        assert result.endswith("user")
        assert result.startswith("*")
