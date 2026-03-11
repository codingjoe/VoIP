"""Tests for SDP message parsing and serialization."""

import pytest
from voip.sdp.messages import SessionDescription
from voip.sdp.types import (
    Attribute,
    Bandwidth,
    ConnectionData,
    MediaDescription,
    Origin,
    RTPPayloadFormat,
    Timing,
)

#: A typical SDP body from a SIP INVITE (RFC 4566 §5 example).
TYPICAL_SDP = (
    b"v=0\r\n"
    b"o=alice 2890844526 2890844526 IN IP4 pc33.atlanta.com\r\n"
    b"s=Session SDP\r\n"
    b"c=IN IP4 224.2.1.1/127/3\r\n"
    b"t=0 0\r\n"
    b"m=audio 49170 RTP/AVP 0\r\n"
    b"a=rtpmap:0 PCMU/8000\r\n"
    b"m=video 51372 RTP/AVP 31\r\n"
    b"a=rtpmap:31 H261/90000\r\n"
    b"m=video 53000 RTP/AVP 32\r\n"
    b"a=rtpmap:32 MPV/90000\r\n"
)

#: A minimal SDP with only required fields.
MINIMAL_SDP = b"v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n"

#: A full-featured SDP exercising optional fields.
FULL_SDP = (
    b"v=0\r\n"
    b"o=jdoe 2890844526 2890844526 IN IP4 jdoe.example.com\r\n"
    b"s=A Seminar on the Web\r\n"
    b"i=A free, world-wide, web seminar\r\n"
    b"u=http://www.example.com/seminars/sdp.ps\r\n"
    b"e=j.doe@example.com (Jane Doe)\r\n"
    b"p=+1 617 555-6011\r\n"
    b"c=IN IP4 224.2.17.12/127\r\n"
    b"b=CT:128\r\n"
    b"t=2873397496 2873404696\r\n"
    b"r=7d 1h 0 25h\r\n"
    b"a=recvonly\r\n"
    b"m=audio 49170 RTP/AVP 0\r\n"
    b"a=rtpmap:0 PCMU/8000\r\n"
    b"m=video 51372 RTP/AVP 99\r\n"
    b"a=rtpmap:99 h263-1998/90000\r\n"
)


class TestSessionDescriptionParse:
    def test_parse__version(self):
        """Parse the version field (v=)."""
        sdp = SessionDescription.parse(
            b"v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\ns=-\r\nt=0 0\r\n"
        )
        assert sdp.version == 0

    def test_parse__origin(self):
        """Parse the origin field (o=)."""
        sdp = SessionDescription.parse(TYPICAL_SDP)
        assert sdp.origin == Origin(
            username="alice",
            sess_id="2890844526",
            sess_version="2890844526",
            nettype="IN",
            addrtype="IP4",
            unicast_address="pc33.atlanta.com",
        )

    def test_parse__session_name(self):
        """Parse the session name field (s=)."""
        sdp = SessionDescription.parse(TYPICAL_SDP)
        assert sdp.name == "Session SDP"

    def test_parse__connection(self):
        """Parse the session-level connection data field (c=)."""
        sdp = SessionDescription.parse(TYPICAL_SDP)
        assert sdp.connection == ConnectionData(
            nettype="IN",
            addrtype="IP4",
            connection_address="224.2.1.1/127/3",
        )

    def test_parse__timing(self):
        """Parse the timing field (t=)."""
        sdp = SessionDescription.parse(TYPICAL_SDP)
        assert sdp.timings == [Timing(start_time=0, stop_time=0)]

    def test_parse__media_count(self):
        """Parse three media sections from the typical SDP."""
        sdp = SessionDescription.parse(TYPICAL_SDP)
        assert len(sdp.media) == 3

    def test_parse__media_audio(self):
        """Parse the audio media section."""
        sdp = SessionDescription.parse(TYPICAL_SDP)
        audio = sdp.media[0]
        assert audio.media == "audio"
        assert audio.port == 49170
        assert audio.proto == "RTP/AVP"
        assert audio.fmt == [
            RTPPayloadFormat(payload_type=0, encoding_name="PCMU", sample_rate=8000)
        ]

    def test_parse__media_video(self):
        """Parse the first video media section."""
        sdp = SessionDescription.parse(TYPICAL_SDP)
        video = sdp.media[1]
        assert video.media == "video"
        assert video.port == 51372

    def test_parse__media_attribute(self):
        """Parse the rtpmap attribute within a media section into RtpPayloadFormat."""
        sdp = SessionDescription.parse(TYPICAL_SDP)
        audio = sdp.media[0]
        assert audio.fmt == [
            RTPPayloadFormat(payload_type=0, encoding_name="PCMU", sample_rate=8000)
        ]
        assert (
            audio.attributes == []
        )  # rtpmap is folded into fmt, not kept as raw attribute

    def test_parse__media_fmtp(self):
        """Parse a=fmtp into the matching RTPPayloadFormat.fmtp field."""
        data = (
            b"v=0\r\n"
            b"o=- 0 0 IN IP4 0.0.0.0\r\n"
            b"s=-\r\n"
            b"t=0 0\r\n"
            b"m=audio 49170 RTP/AVP 111\r\n"
            b"a=rtpmap:111 opus/48000/2\r\n"
            b"a=fmtp:111 minptime=10;useinbandfec=1\r\n"
        )
        sdp = SessionDescription.parse(data)
        audio = sdp.media[0]
        assert audio.fmt == [
            RTPPayloadFormat(
                payload_type=111,
                encoding_name="opus",
                sample_rate=48000,
                channels=2,
                fmtp="minptime=10;useinbandfec=1",
            )
        ]
        assert (
            audio.attributes == []
        )  # fmtp is folded into fmt, not kept as raw attribute

    def test_parse__title(self):
        """Parse the session title field (i=) at session level."""
        sdp = SessionDescription.parse(FULL_SDP)
        assert sdp.title == "A free, world-wide, web seminar"

    def test_parse__uri(self):
        """Parse the URI field (u=)."""
        sdp = SessionDescription.parse(FULL_SDP)
        assert sdp.uri == "http://www.example.com/seminars/sdp.ps"

    def test_parse__email(self):
        """Parse the email field (e=)."""
        sdp = SessionDescription.parse(FULL_SDP)
        assert sdp.emails == ["j.doe@example.com (Jane Doe)"]

    def test_parse__phone(self):
        """Parse the phone field (p=)."""
        sdp = SessionDescription.parse(FULL_SDP)
        assert sdp.phones == ["+1 617 555-6011"]

    def test_parse__bandwidth(self):
        """Parse the bandwidth field (b=) at session level."""
        sdp = SessionDescription.parse(FULL_SDP)
        assert sdp.bandwidths == [Bandwidth(bwtype="CT", bandwidth=128)]

    def test_parse__repeat(self):
        """Parse the repeat times field (r=)."""
        sdp = SessionDescription.parse(FULL_SDP)
        assert sdp.repeat == "7d 1h 0 25h"

    def test_parse__session_attribute(self):
        """Parse the session-level attribute (a=)."""
        sdp = SessionDescription.parse(FULL_SDP)
        assert Attribute(name="recvonly") in sdp.attributes

    def test_parse__media_title(self):
        """Parse the media-level title (i=) belonging to a media section."""
        data = (
            b"v=0\r\n"
            b"o=- 0 0 IN IP4 0.0.0.0\r\n"
            b"s=-\r\n"
            b"t=0 0\r\n"
            b"m=audio 49170 RTP/AVP 0\r\n"
            b"i=Audio stream\r\n"
        )
        sdp = SessionDescription.parse(data)
        assert sdp.media[0].title == "Audio stream"

    def test_parse__media_connection(self):
        """Parse the media-level connection data (c=) belonging to a media section."""
        data = (
            b"v=0\r\n"
            b"o=- 0 0 IN IP4 0.0.0.0\r\n"
            b"s=-\r\n"
            b"t=0 0\r\n"
            b"m=audio 49170 RTP/AVP 0\r\n"
            b"c=IN IP4 192.168.1.1\r\n"
        )
        sdp = SessionDescription.parse(data)
        assert sdp.media[0].connection == ConnectionData(
            nettype="IN", addrtype="IP4", connection_address="192.168.1.1"
        )

    def test_parse__media_bandwidth(self):
        """Parse the media-level bandwidth (b=) belonging to a media section."""
        data = (
            b"v=0\r\n"
            b"o=- 0 0 IN IP4 0.0.0.0\r\n"
            b"s=-\r\n"
            b"t=0 0\r\n"
            b"m=audio 49170 RTP/AVP 0\r\n"
            b"b=AS:64\r\n"
        )
        sdp = SessionDescription.parse(data)
        assert sdp.media[0].bandwidths == [Bandwidth(bwtype="AS", bandwidth=64)]

    def test_parse__str_input(self):
        """Accept a str input in addition to bytes."""
        text = TYPICAL_SDP.decode()
        sdp = SessionDescription.parse(text)
        assert sdp.version == 0
        assert len(sdp.media) == 3

    def test_parse__skips_invalid_lines(self):
        """Skip lines without an = separator."""
        data = b"v=0\r\nINVALID\r\no=- 0 0 IN IP4 0.0.0.0\r\ns=-\r\nt=0 0\r\n"
        sdp = SessionDescription.parse(data)
        assert sdp.version == 0

    def test_parse__multiple_timings(self):
        """Parse multiple timing fields (t=)."""
        data = b"v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\ns=-\r\nt=0 0\r\nt=100 200\r\n"
        sdp = SessionDescription.parse(data)
        assert sdp.timings == [
            Timing(start_time=0, stop_time=0),
            Timing(start_time=100, stop_time=200),
        ]

    def test_parse__zone(self):
        """Parse the time zone adjustment field (z=)."""
        data = (
            b"v=0\r\n"
            b"o=- 0 0 IN IP4 0.0.0.0\r\n"
            b"s=-\r\n"
            b"t=2882844526 2898848070\r\n"
            b"z=2882844526 -1h 2898848070 0\r\n"
        )
        sdp = SessionDescription.parse(data)
        assert sdp.zone == "2882844526 -1h 2898848070 0"


class TestSessionDescriptionStr:
    def test_str__version(self):
        """Serialize the version field."""
        sdp = SessionDescription(version=0)
        assert "v=0\r\n" in str(sdp)

    def test_str__origin(self):
        """Serialize the origin field."""
        sdp = SessionDescription(
            origin=Origin(
                username="alice",
                sess_id="2890844526",
                sess_version="2890844526",
                nettype="IN",
                addrtype="IP4",
                unicast_address="pc33.atlanta.com",
            )
        )
        assert "o=alice 2890844526 2890844526 IN IP4 pc33.atlanta.com\r\n" in str(sdp)

    def test_str__session_name(self):
        """Serialize the session name field."""
        sdp = SessionDescription(name="My Session")
        assert "s=My Session\r\n" in str(sdp)

    def test_str__no_origin(self):
        """Omit the origin line when origin is None."""
        sdp = SessionDescription()
        assert "o=" not in str(sdp)

    def test_str__connection(self):
        """Serialize the connection data field."""
        sdp = SessionDescription(
            connection=ConnectionData(
                nettype="IN", addrtype="IP4", connection_address="192.168.1.1"
            )
        )
        assert "c=IN IP4 192.168.1.1\r\n" in str(sdp)

    def test_str__title(self):
        """Serialize the session title field."""
        sdp = SessionDescription(title="Test Title")
        assert "i=Test Title\r\n" in str(sdp)

    def test_str__uri(self):
        """Serialize the URI field."""
        sdp = SessionDescription(uri="http://example.com")
        assert "u=http://example.com\r\n" in str(sdp)

    def test_str__emails(self):
        """Serialize email fields."""
        sdp = SessionDescription(emails=["user@example.com"])
        assert "e=user@example.com\r\n" in str(sdp)

    def test_str__phones(self):
        """Serialize phone fields."""
        sdp = SessionDescription(phones=["+1 555 1234"])
        assert "p=+1 555 1234\r\n" in str(sdp)

    def test_str__bandwidth(self):
        """Serialize the bandwidth field."""
        sdp = SessionDescription(bandwidths=[Bandwidth(bwtype="CT", bandwidth=128)])
        assert "b=CT:128\r\n" in str(sdp)

    def test_str__timing(self):
        """Serialize the timing field."""
        sdp = SessionDescription(timings=[Timing(start_time=0, stop_time=0)])
        assert "t=0 0\r\n" in str(sdp)

    def test_str__repeat(self):
        """Serialize the repeat times field."""
        sdp = SessionDescription(
            timings=[Timing(start_time=0, stop_time=0)], repeat="7d 1h 0 25h"
        )
        assert "r=7d 1h 0 25h\r\n" in str(sdp)

    def test_str__zone(self):
        """Serialize the time zone field."""
        sdp = SessionDescription(zone="2882844526 -1h")
        assert "z=2882844526 -1h\r\n" in str(sdp)

    def test_str__attribute_flag(self):
        """Serialize a flag-style attribute (no value)."""
        sdp = SessionDescription(attributes=[Attribute(name="recvonly")])
        assert "a=recvonly\r\n" in str(sdp)

    def test_str__attribute_value(self):
        """Serialize an attribute with a value."""
        sdp = SessionDescription(
            attributes=[Attribute(name="rtpmap", value="0 PCMU/8000")]
        )
        assert "a=rtpmap:0 PCMU/8000\r\n" in str(sdp)

    def test_str__media(self):
        """Serialize a media section."""
        sdp = SessionDescription(
            media=[
                MediaDescription(
                    media="audio",
                    port=49170,
                    proto="RTP/AVP",
                    fmt=[
                        RTPPayloadFormat(
                            payload_type=0, encoding_name="PCMU", sample_rate=8000
                        )
                    ],
                )
            ]
        )
        text = str(sdp)
        assert "m=audio 49170 RTP/AVP 0\r\n" in text
        assert "a=rtpmap:0 PCMU/8000\r\n" in text

    def test_str__media_with_connection(self):
        """Serialize a media section with connection data."""
        sdp = SessionDescription(
            media=[
                MediaDescription(
                    media="audio",
                    port=49170,
                    proto="RTP/AVP",
                    fmt=[RTPPayloadFormat(payload_type=0)],
                    connection=ConnectionData(
                        nettype="IN", addrtype="IP4", connection_address="192.168.1.1"
                    ),
                )
            ]
        )
        assert "c=IN IP4 192.168.1.1\r\n" in str(sdp)

    def test_str__media_with_title(self):
        """Serialize a media section with a title."""
        sdp = SessionDescription(
            media=[
                MediaDescription(
                    media="audio",
                    port=49170,
                    proto="RTP/AVP",
                    fmt=[RTPPayloadFormat(payload_type=0)],
                    title="My Audio",
                )
            ]
        )
        assert "i=My Audio\r\n" in str(sdp)

    def test_str__ends_with_crlf(self):
        """Ensure serialized SDP ends with CRLF."""
        sdp = SessionDescription()
        assert str(sdp).endswith("\r\n")


class TestSessionDescriptionBytes:
    def test_bytes__returns_bytes(self):
        """Return bytes from __bytes__."""
        sdp = SessionDescription()
        assert isinstance(bytes(sdp), bytes)

    def test_bytes__matches_str(self):
        """Match bytes to the encoded str representation."""
        sdp = SessionDescription(
            version=0,
            origin=Origin(
                username="-",
                sess_id="0",
                sess_version="0",
                nettype="IN",
                addrtype="IP4",
                unicast_address="127.0.0.1",
            ),
            name="-",
            timings=[Timing(start_time=0, stop_time=0)],
        )
        assert bytes(sdp) == str(sdp).encode()


class TestSessionDescriptionRoundtrip:
    def test_roundtrip__typical(self):
        """Round-trip a typical SDP through parse and serialization."""
        sdp = SessionDescription.parse(TYPICAL_SDP)
        result = bytes(sdp)
        assert result == TYPICAL_SDP

    def test_roundtrip__minimal(self):
        """Round-trip a minimal SDP through parse and serialization."""
        sdp = SessionDescription.parse(MINIMAL_SDP)
        assert bytes(sdp) == MINIMAL_SDP

    def test_roundtrip__full(self):
        """Round-trip the full-featured SDP through parse and serialization."""
        sdp = SessionDescription.parse(FULL_SDP)
        assert bytes(sdp) == FULL_SDP


class TestOrigin:
    def test_parse(self):
        """Parse an o= field value."""
        origin = Origin.parse("alice 123 456 IN IP4 host.example.com")
        assert origin == Origin(
            username="alice",
            sess_id="123",
            sess_version="456",
            nettype="IN",
            addrtype="IP4",
            unicast_address="host.example.com",
        )

    def test_str(self):
        """Serialize an Origin to string."""
        origin = Origin(
            username="alice",
            sess_id="123",
            sess_version="456",
            nettype="IN",
            addrtype="IP4",
            unicast_address="host.example.com",
        )
        assert str(origin) == "alice 123 456 IN IP4 host.example.com"


class TestConnectionData:
    def test_parse(self):
        """Parse a c= field value."""
        conn = ConnectionData.parse("IN IP4 224.2.1.1/127/3")
        assert conn == ConnectionData(
            nettype="IN", addrtype="IP4", connection_address="224.2.1.1/127/3"
        )

    def test_str(self):
        """Serialize ConnectionData to string."""
        conn = ConnectionData(
            nettype="IN", addrtype="IP4", connection_address="192.0.2.1"
        )
        assert str(conn) == "IN IP4 192.0.2.1"


class TestBandwidth:
    def test_parse(self):
        """Parse a b= field value."""
        bw = Bandwidth.parse("AS:128")
        assert bw == Bandwidth(bwtype="AS", bandwidth=128)

    def test_str(self):
        """Serialize Bandwidth to string."""
        bw = Bandwidth(bwtype="CT", bandwidth=512)
        assert str(bw) == "CT:512"


class TestTiming:
    def test_parse(self):
        """Parse a t= field value."""
        timing = Timing.parse("2873397496 2873404696")
        assert timing == Timing(start_time=2873397496, stop_time=2873404696)

    def test_str(self):
        """Serialize Timing to string."""
        timing = Timing(start_time=0, stop_time=0)
        assert str(timing) == "0 0"


class TestAttribute:
    def test_parse__flag(self):
        """Parse a flag attribute (no value)."""
        attr = Attribute.parse("recvonly")
        assert attr == Attribute(name="recvonly", value=None)

    def test_parse__value(self):
        """Parse an attribute with a value."""
        attr = Attribute.parse("rtpmap:96 opus/48000/2")
        assert attr == Attribute(name="rtpmap", value="96 opus/48000/2")

    def test_str__flag(self):
        """Serialize a flag attribute."""
        assert str(Attribute(name="recvonly")) == "recvonly"

    def test_str__value(self):
        """Serialize an attribute with a value."""
        assert (
            str(Attribute(name="rtpmap", value="96 opus/48000/2"))
            == "rtpmap:96 opus/48000/2"
        )


class TestMediaDescription:
    def test_parse(self):
        """Parse an m= field value."""
        media = MediaDescription.parse("audio 49170 RTP/AVP 0 8")
        assert media == MediaDescription(
            media="audio",
            port=49170,
            proto="RTP/AVP",
            fmt=[RTPPayloadFormat(payload_type=0), RTPPayloadFormat(payload_type=8)],
        )

    def test_str(self):
        """Serialize a MediaDescription; static PTs include a=rtpmap with RFC 3551 defaults."""
        media = MediaDescription(
            media="audio",
            port=49170,
            proto="RTP/AVP",
            fmt=[RTPPayloadFormat(payload_type=0)],
        )
        assert str(media) == "m=audio 49170 RTP/AVP 0\r\na=rtpmap:0 PCMU/8000"

    def test_str__with_rtpmap_in_fmt(self):
        """Serialize a MediaDescription; a=rtpmap is derived from fmt entries with codec info."""
        media = MediaDescription(
            media="audio",
            port=49170,
            proto="RTP/AVP",
            fmt=[
                RTPPayloadFormat(payload_type=0, encoding_name="PCMU", sample_rate=8000)
            ],
        )
        assert str(media) == "m=audio 49170 RTP/AVP 0\r\na=rtpmap:0 PCMU/8000"

    def test_str__with_fmtp_in_fmt(self):
        """Serialize a MediaDescription; a=fmtp is emitted after a=rtpmap when fmtp is set."""
        media = MediaDescription(
            media="audio",
            port=49170,
            proto="RTP/AVP",
            fmt=[
                RTPPayloadFormat(
                    payload_type=111,
                    encoding_name="opus",
                    sample_rate=48000,
                    channels=2,
                    fmtp="minptime=10;useinbandfec=1",
                )
            ],
        )
        assert str(media) == (
            "m=audio 49170 RTP/AVP 111\r\n"
            "a=rtpmap:111 opus/48000/2\r\n"
            "a=fmtp:111 minptime=10;useinbandfec=1"
        )

    def test_parse__with_rtpmap(self):
        """Parse a multi-line block; a=rtpmap populates encoding_name, sample_rate."""
        media = MediaDescription.parse(
            "audio 49170 RTP/AVP 111\r\na=rtpmap:111 opus/48000/2"
        )
        assert media.fmt[0].encoding_name == "opus"
        assert media.fmt[0].sample_rate == 48000
        assert media.fmt[0].channels == 2

    def test_parse__with_fmtp(self):
        """Parse a multi-line block; a=fmtp populates fmtp on the matching format."""
        media = MediaDescription.parse(
            "audio 49170 RTP/AVP 111\r\n"
            "a=rtpmap:111 opus/48000/2\r\n"
            "a=fmtp:111 minptime=10;useinbandfec=1"
        )
        assert media.fmt[0].fmtp == "minptime=10;useinbandfec=1"

    def test_parse__roundtrip(self):
        """str() output round-trips through parse() (with m= prefix)."""
        original = MediaDescription(
            media="audio",
            port=49170,
            proto="RTP/AVP",
            fmt=[
                RTPPayloadFormat(
                    payload_type=111,
                    encoding_name="opus",
                    sample_rate=48000,
                    channels=2,
                    fmtp="minptime=10;useinbandfec=1",
                )
            ],
        )
        assert MediaDescription.parse(str(original)) == original

    def test_parse__generic_attribute_preserved(self):
        """Non-rtpmap/fmtp a= lines are added to the attributes list."""
        media = MediaDescription.parse("audio 49170 RTP/AVP 0\r\na=sendrecv")
        assert media.attributes == [Attribute(name="sendrecv", value=None)]

    def test_parse__multiple_formats(self):
        """a=rtpmap lines are matched to the correct payload type."""
        media = MediaDescription.parse(
            "audio 49170 RTP/AVP 0 111\r\n"
            "a=rtpmap:111 opus/48000/2\r\n"
            "a=fmtp:111 minptime=10"
        )
        pcmu = media.get_format(0)
        opus = media.get_format(111)
        assert pcmu is not None and pcmu.encoding_name == "PCMU"
        assert opus is not None and opus.encoding_name == "opus"
        assert opus.channels == 2
        assert opus.fmtp == "minptime=10"


class TestRTPPayloadFormat:
    def test_parse__opus(self):
        """Parse a full Opus rtpmap attribute value."""
        rm = RTPPayloadFormat.parse("111 opus/48000/2")
        assert rm.payload_type == 111
        assert rm.encoding_name == "opus"
        assert rm.sample_rate == 48000
        assert rm.channels == 2

    def test_parse__pcma(self):
        """Parse a PCMA rtpmap attribute value (no channel count)."""
        rm = RTPPayloadFormat.parse("8 PCMA/8000")
        assert rm.payload_type == 8
        assert rm.encoding_name == "PCMA"
        assert rm.sample_rate == 8000
        assert rm.channels == 1

    def test_parse__invalid__raises(self):
        """Raise ValueError for a malformed rtpmap value."""
        with pytest.raises(ValueError):
            RTPPayloadFormat.parse("111 opus")

    def test_str__with_channels(self):
        """Serialize an RtpPayloadFormat with channel count > 1."""
        rm = RTPPayloadFormat(
            payload_type=111, encoding_name="opus", sample_rate=48000, channels=2
        )
        assert str(rm) == "111 opus/48000/2"

    def test_str__without_channels(self):
        """Serialize an RtpPayloadFormat with a single channel (omit channel suffix)."""
        rm = RTPPayloadFormat(payload_type=8, encoding_name="PCMA", sample_rate=8000)
        assert str(rm) == "8 PCMA/8000"


class TestMediaDescriptionGetFormat:
    def test_get_format__found(self):
        """Return the RtpPayloadFormat for a known payload type."""
        media = MediaDescription(
            media="audio",
            port=0,
            proto="RTP/AVP",
            fmt=[
                RTPPayloadFormat(
                    payload_type=111,
                    encoding_name="opus",
                    sample_rate=48000,
                    channels=2,
                )
            ],
        )
        f = media.get_format(111)
        assert f is not None
        assert f.encoding_name == "opus"

    def test_get_format__found_by_str(self):
        """Accept a string payload type in get_format."""
        media = MediaDescription(
            media="audio",
            port=0,
            proto="RTP/AVP",
            fmt=[
                RTPPayloadFormat(payload_type=8, encoding_name="PCMA", sample_rate=8000)
            ],
        )
        assert media.get_format("8") is not None

    def test_get_format__not_found(self):
        """Return None when no format matches the given payload type."""
        media = MediaDescription(
            media="audio",
            port=0,
            proto="RTP/AVP",
            fmt=[RTPPayloadFormat(payload_type=8)],
        )
        assert media.get_format(111) is None


class TestMediaDescriptionSampleRate:
    def test_sample_rate__from_rtpmap(self):
        """Return the sample rate from an explicit a=rtpmap attribute."""
        f = RTPPayloadFormat(
            payload_type=111, encoding_name="opus", sample_rate=48000, channels=2
        )
        assert f.sample_rate == 48000

    def test_sample_rate__static_pcmu(self):
        """Return 8000 Hz for PCMU (PT 0) via StaticPayloadType fallback."""
        f = RTPPayloadFormat(payload_type=0)
        assert f.sample_rate == 8000

    def test_sample_rate__static_pcma(self):
        """Return 8000 Hz for PCMA (PT 8) via StaticPayloadType fallback."""
        f = RTPPayloadFormat(payload_type=8)
        assert f.sample_rate == 8000

    def test_sample_rate__static_g722(self):
        """Return 8000 Hz for G.722 (PT 9) via StaticPayloadType fallback."""
        f = RTPPayloadFormat(payload_type=9)
        assert f.sample_rate == 8000

    def test_sample_rate__unknown_dynamic_pt__returns_none(self):
        """Dynamic PTs without an a=rtpmap have sample_rate=None (no RFC 3551 default)."""
        f = RTPPayloadFormat(payload_type=99)
        assert f.sample_rate is None
