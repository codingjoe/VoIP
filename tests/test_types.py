import ipaddress

import pytest
from voip.types import NetworkAddress


class TestNetworkAddress:
    @pytest.mark.parametrize(
        "address,expected",
        [
            # domain
            (NetworkAddress("example.com"), "example.com"),
            (NetworkAddress("example.com", 80), "example.com:80"),
            # IPv4
            (NetworkAddress(ipaddress.IPv4Address("127.0.0.1")), "127.0.0.1"),
            (NetworkAddress(ipaddress.IPv4Address("127.0.0.1"), 80), "127.0.0.1:80"),
            # IPv6
            (NetworkAddress(ipaddress.IPv6Address("2001:db8::1")), "2001:db8::1"),
            (
                NetworkAddress(ipaddress.IPv6Address("2001:db8::1"), 80),
                "[2001:db8::1]:80",
            ),
        ],
    )
    def test_str(self, address, expected):
        assert str(address) == expected

    @pytest.mark.parametrize(
        "data,expected",
        [
            # domain
            ("example.com", NetworkAddress("example.com")),
            ("example.com:80", NetworkAddress("example.com", 80)),
            # IPv4
            ("127.0.0.1", NetworkAddress(ipaddress.IPv4Address("127.0.0.1"))),
            ("127.0.0.1:80", NetworkAddress(ipaddress.IPv4Address("127.0.0.1"), 80)),
            # IPv6
            ("2001:db8::1", NetworkAddress(ipaddress.IPv6Address("2001:db8::1"))),
            (
                "[2001:db8::1]:80",
                NetworkAddress(ipaddress.IPv6Address("2001:db8::1"), 80),
            ),
        ],
    )
    def test_parse(self, data, expected):
        assert NetworkAddress.parse(data) == expected

    def test_parse__value_error__host(self):
        with pytest.raises(ValueError) as exc_info:
            NetworkAddress.parse("example.com:invalid_port")
        assert (
            str(exc_info.value) == "Invalid network address: 'example.com:invalid_port'"
        )
