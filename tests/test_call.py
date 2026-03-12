"""Tests for the SIP session and call handler hierarchy."""

import asyncio
from unittest.mock import MagicMock

import pytest
from voip.audio import AudioCall
from voip.rtp import RealtimeTransportProtocol
from voip.sip.messages import Message, Request, Response
from voip.sip.protocol import SIP, SessionInitiationProtocol


def make_audio_call(**kwargs) -> AudioCall:
    """Create an AudioCall with mock rtp/sip for unit testing."""
    defaults: dict = {
        "rtp": MagicMock(spec=RealtimeTransportProtocol),
        "sip": MagicMock(),
    }
    defaults.update(kwargs)
    return AudioCall(**defaults)


class TestAudioCall:
    def test_caller__returns_caller_arg(self):
        """Return the caller string passed at construction."""
        call = make_audio_call(caller="sip:bob@biloxi.com")
        assert call.caller == "sip:bob@biloxi.com"

    def test_caller__defaults_to_empty_string(self):
        """Return an empty string when no caller is given."""
        assert make_audio_call().caller == ""

    def test_audio_received__noop_by_default(self):
        """audio_received is a no-op in the base AudioCall class."""
        sentinel = object()
        make_audio_call().audio_received(sentinel)  # must not raise

    def test_rtp_and_sip_stored_as_fields(self):
        """Rtp and sip back-references are stored as dataclass fields."""
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_sip = MagicMock()
        call = AudioCall(rtp=mock_rtp, sip=mock_sip)
        assert call.rtp is mock_rtp
        assert call.sip is mock_sip

    def test_init__stores_media(self):
        """Media parameter is stored on the AudioCall instance."""
        from voip.sdp.types import MediaDescription, RTPPayloadFormat  # noqa: PLC0415

        media = MediaDescription(
            media="audio",
            port=49170,
            proto="RTP/AVP",
            fmt=[
                RTPPayloadFormat(payload_type=8, encoding_name="PCMA", sample_rate=8000)
            ],
        )
        call = make_audio_call(media=media)
        assert call.media is media

    def test_init__derives_sample_rate_from_media(self):
        """sample_rate is derived from the RTPPayloadFormat sample_rate."""
        from voip.sdp.types import MediaDescription, RTPPayloadFormat  # noqa: PLC0415

        media = MediaDescription(
            media="audio",
            port=49170,
            proto="RTP/AVP",
            fmt=[
                RTPPayloadFormat(payload_type=9, encoding_name="G722", sample_rate=8000)
            ],
        )
        call = make_audio_call(media=media)
        assert call.sample_rate == 8000

    def test_init__default_sample_rate_without_media(self):
        """Default sample_rate is 8000 Hz when no MediaDescription is given."""
        assert make_audio_call().sample_rate == 8000

    def test_init__derives_payload_type_from_media(self):
        """payload_type is derived from the first fmt entry of the MediaDescription."""
        from voip.sdp.types import MediaDescription, RTPPayloadFormat  # noqa: PLC0415

        media = MediaDescription(
            media="audio",
            port=49170,
            proto="RTP/AVP",
            fmt=[RTPPayloadFormat(payload_type=8)],
        )
        call = make_audio_call(media=media)
        assert call.payload_type == 8

    def test_init__default_payload_type_without_media(self):
        """Default payload_type is 0 when no MediaDescription is given."""
        assert make_audio_call().payload_type == 0

    def test_init__logs_codec_info(self, caplog):
        """Log codec name, sample rate and payload type at INFO level on init."""
        import logging  # noqa: PLC0415

        from voip.sdp.types import MediaDescription, RTPPayloadFormat  # noqa: PLC0415

        media = MediaDescription(
            media="audio",
            port=49170,
            proto="RTP/AVP",
            fmt=[
                RTPPayloadFormat(payload_type=8, encoding_name="PCMA", sample_rate=8000)
            ],
        )
        with caplog.at_level(logging.INFO, logger="voip.audio"):
            make_audio_call(media=media)
        assert any("PCMA" in r.message and "8000" in r.message for r in caplog.records)

    def test_chunk_duration__default_is_zero(self):
        """chunk_duration defaults to 0 (per-packet mode)."""
        assert AudioCall.chunk_duration == 0

    @pytest.mark.asyncio
    async def test_datagram_received__forwards_audio_payload(self):
        """datagram_received parses the RTP packet and calls audio_received after decode."""
        import struct  # noqa: PLC0415

        DECODED = object()  # sentinel: returned by mocked _decode_raw
        received: list = []

        class ConcreteCall(AudioCall):
            def _decode_raw(self, raw_packets: list[bytes]):
                return DECODED  # skip real av decode in unit tests

            def audio_received(self, audio) -> None:
                received.append(audio)

        rtp_packet = struct.pack(">BBHII", 0x80, 111 & 0x7F, 1, 0, 0) + b"audio"
        call = ConcreteCall(rtp=MagicMock(), sip=MagicMock())
        call.datagram_received(rtp_packet, ("127.0.0.1", 5004))
        await asyncio.sleep(0.05)  # let the executor task run
        assert received == [DECODED]

    @pytest.mark.asyncio
    async def test_datagram_received__chunk_duration__buffers_until_threshold(self):
        """Buffer packets and emit audio_received only when the threshold is reached."""
        import struct  # noqa: PLC0415

        from voip.sdp.types import MediaDescription, RTPPayloadFormat  # noqa: PLC0415

        received: list = []
        media = MediaDescription(
            media="audio",
            port=0,
            proto="RTP/AVP",
            fmt=[
                RTPPayloadFormat(payload_type=8, encoding_name="PCMA", sample_rate=8000)
            ],
        )

        class ChunkedCall(AudioCall):
            chunk_duration = 1  # 1 s @ 8 kHz / 160 samples = 50 packets

            def _decode_raw(self, raw_packets: list[bytes]):
                return raw_packets  # skip real av decode; pass raw list as "audio"

            def audio_received(self, audio) -> None:
                received.append(audio)

        call = ChunkedCall(rtp=MagicMock(), sip=MagicMock(), media=media)
        assert call._packet_threshold == 50
        rtp_packet = struct.pack(">BBHII", 0x80, 8 & 0x7F, 1, 0, 0) + b"x"
        for _ in range(49):
            call.datagram_received(rtp_packet, ("127.0.0.1", 5004))
        assert len(call._audio_buffer) == 49
        assert received == []  # threshold not yet reached
        call.datagram_received(
            struct.pack(">BBHII", 0x80, 8 & 0x7F, 2, 0, 0) + b"last",
            ("127.0.0.1", 5004),
        )
        assert len(call._audio_buffer) == 0  # buffer drained
        await asyncio.sleep(0.05)  # let the executor task run
        assert len(received) == 1
        assert len(received[0]) == 50


class TestNegotiateCodec:
    def _make_media(self, fmts: list[str], rtpmaps: list[str] | None = None):
        """Build a MediaDescription with given format list and optional rtpmap attributes."""
        from voip.sdp.types import MediaDescription, RTPPayloadFormat  # noqa: PLC0415

        rtpmap_by_pt: dict[int, RTPPayloadFormat] = {}
        for rtpmap in rtpmaps or []:
            f = RTPPayloadFormat.parse(rtpmap)
            rtpmap_by_pt[f.payload_type] = f
        formats = [
            rtpmap_by_pt.get(int(pt)) or RTPPayloadFormat(payload_type=int(pt))
            for pt in fmts
        ]
        return MediaDescription(media="audio", port=49170, proto="RTP/AVP", fmt=formats)

    def test_negotiate_codec__prefers_opus(self):
        """Select Opus when offered alongside lower-priority codecs."""
        media = self._make_media(["0", "8", "111"], ["111 opus/48000/2", "8 PCMA/8000"])
        result = AudioCall.negotiate_codec(media)
        assert result.fmt[0].payload_type == 111
        assert result.fmt[0].sample_rate == 48000

    def test_negotiate_codec__falls_back_to_pcma(self):
        """Select PCMA when Opus and G.722 are not offered."""
        media = self._make_media(["0", "8"])
        result = AudioCall.negotiate_codec(media)
        assert result.fmt[0].payload_type == 8
        assert result.fmt[0].sample_rate == 8000

    def test_negotiate_codec__falls_back_to_pcmu(self):
        """Select PCMU when only PCMU is offered."""
        media = self._make_media(["0"])
        result = AudioCall.negotiate_codec(media)
        assert result.fmt[0].payload_type == 0

    def test_negotiate_codec__empty_fmt__raises(self):
        """Raise NotImplementedError when the remote side offers no audio formats."""
        media = self._make_media([])
        with pytest.raises(NotImplementedError):
            AudioCall.negotiate_codec(media)

    def test_negotiate_codec__unknown_codec__raises(self):
        """Raise NotImplementedError when no offered codec matches PREFERRED_CODECS."""
        media = self._make_media(["126"], ["126 telephone-event/8000"])
        with pytest.raises(NotImplementedError):
            AudioCall.negotiate_codec(media)

    def test_negotiate_codec__returns_media_description(self):
        """negotiate_codec returns a MediaDescription object."""
        from voip.sdp.types import MediaDescription  # noqa: PLC0415

        media = self._make_media(["0", "8", "111"], ["111 opus/48000/2"])
        result = AudioCall.negotiate_codec(media)
        assert isinstance(result, MediaDescription)
        assert result.media == "audio"
        assert result.proto == "RTP/AVP"

    def test_negotiate_codec__subclass_can_override_preferences(self):
        """A subclass with a different PREFERRED_CODECS list uses its own preferences."""
        from voip.sdp.types import RTPPayloadFormat  # noqa: PLC0415

        class PCMAOnlyCall(AudioCall):
            PREFERRED_CODECS = [
                RTPPayloadFormat(payload_type=8, encoding_name="PCMA", sample_rate=8000)
            ]

        media = self._make_media(["0", "8", "111"])
        result = PCMAOnlyCall.negotiate_codec(media)
        assert result.fmt[0].payload_type == 8


class TestSIP:
    class _CapturingSIP(SIP):
        """SIP subclass that captures sent messages without monkey-patching slots."""

        def __init__(self):
            super().__init__(
                server_address=("127.0.0.1", 5060),
                aor="sip:test@example.com",
            )
            self._sent: list[tuple] = []

        def send(self, message, addr):
            self._sent.append((message, addr))

    async def test_connection_made__stores_transport(self):
        """Store the transport when a connection is established."""
        protocol = SIP(server_address=("127.0.0.1", 5060), aor="sip:test@example.com")
        transport = MagicMock()
        protocol.connection_made(transport)
        assert protocol.transport is transport

    async def test_send__serializes_and_forwards_to_transport(self):
        """Serialize the message and forward it to the underlying transport."""
        protocol = SIP(server_address=("127.0.0.1", 5060), aor="sip:test@example.com")
        transport = MagicMock()
        protocol.connection_made(transport)
        transport.sendto.reset_mock()  # clear the STUN request call
        response = Response(status_code=200, reason="OK")
        addr = ("192.0.2.1", 5060)
        protocol.send(response, addr)
        protocol.transport.sendto.assert_called_once_with(bytes(response), addr)

    async def test_request_received__invite__stores_addr_and_calls_call_received(self):
        """Dispatch an INVITE to call_received and store the addr by Call-ID."""
        received = []

        class MySIP(SIP):
            def call_received(self, request):
                received.append(request)

        protocol = MySIP(server_address=("127.0.0.1", 5060), aor="sip:test@example.com")
        protocol.connection_made(MagicMock())
        request = make_invite()
        addr = ("192.0.2.1", 5060)
        protocol.request_received(request, addr)
        assert len(received) == 1
        assert received[0] is request
        assert protocol._request_addrs.get(request.headers["Call-ID"]) == addr

    async def test_call_received__noop_by_default(self):
        """call_received is a no-op in the base class."""
        protocol = SIP(server_address=("127.0.0.1", 5060), aor="sip:test@example.com")
        protocol.connection_made(MagicMock())
        protocol.call_received(make_invite())  # must not raise

    async def test_answer__sends_200_ok(self):
        """Send a 200 OK response with an SDP body when answering."""
        loop = asyncio.get_running_loop()
        protocol = self._CapturingSIP()
        protocol.transport = make_mock_transport()
        protocol.public_address = loop.create_future()
        protocol.public_address.set_result(("127.0.0.1", 5060))
        mux = RealtimeTransportProtocol()
        mux.public_address = loop.create_future()
        mux.public_address.set_result(("127.0.0.1", 12000))
        mock_rtp_transport = MagicMock()
        mock_rtp_transport.get_extra_info.return_value = ("127.0.0.1", 12000)
        protocol._rtp_protocol = mux
        protocol._rtp_transport = mock_rtp_transport
        request = make_invite()
        protocol._request_addrs[request.headers["Call-ID"]] = ("192.0.2.1", 5060)
        await protocol._answer(request, AudioCall)
        assert len(protocol._sent) == 1
        response, addr = protocol._sent[0]
        assert response.status_code == 200
        assert response.reason == "OK"
        assert addr == ("192.0.2.1", 5060)

    async def test_answer__sdp_contains_opus_audio_line(self):
        """Include an audio media line in the SDP body of the 200 OK."""
        loop = asyncio.get_running_loop()
        protocol = self._CapturingSIP()
        protocol.transport = make_mock_transport()
        protocol.public_address = loop.create_future()
        protocol.public_address.set_result(("127.0.0.1", 5060))
        mux = RealtimeTransportProtocol()
        mux.public_address = loop.create_future()
        mux.public_address.set_result(("127.0.0.1", 12000))
        mock_rtp_transport = MagicMock()
        mock_rtp_transport.get_extra_info.return_value = ("127.0.0.1", 12000)
        protocol._rtp_protocol = mux
        protocol._rtp_transport = mock_rtp_transport
        request = make_invite()
        protocol._request_addrs[request.headers["Call-ID"]] = ("192.0.2.1", 5060)
        await protocol._answer(request, AudioCall)
        response, _ = protocol._sent[0]
        assert b"m=audio" in bytes(response.body)
        assert b"RTP/AVP 0" in bytes(response.body)

    async def test_answer__copies_dialog_headers(self):
        """Copy Via, To, From, Call-ID, and CSeq headers into the 200 OK."""
        loop = asyncio.get_running_loop()
        protocol = self._CapturingSIP()
        protocol.transport = make_mock_transport()
        protocol.public_address = loop.create_future()
        protocol.public_address.set_result(("127.0.0.1", 5060))
        mux = RealtimeTransportProtocol()
        mux.public_address = loop.create_future()
        mux.public_address.set_result(("127.0.0.1", 12000))
        mock_rtp_transport = MagicMock()
        mock_rtp_transport.get_extra_info.return_value = ("127.0.0.1", 12000)
        protocol._rtp_protocol = mux
        protocol._rtp_transport = mock_rtp_transport
        request = make_invite()
        protocol._request_addrs[request.headers["Call-ID"]] = ("192.0.2.1", 5060)
        await protocol._answer(request, AudioCall)
        response, _ = protocol._sent[0]
        assert response.headers["Via"] == "SIP/2.0/UDP pc33.atlanta.com"
        assert response.headers["To"] == "sip:alice@atlanta.com"
        assert response.headers["From"] == "sip:bob@biloxi.com"
        assert response.headers["Call-ID"] == "1234@pc33"
        assert response.headers["CSeq"] == "1 INVITE"

    async def test_answer__instantiates_call_class_with_caller(self):
        """The call_class is instantiated with the caller from the From header."""
        created: list[str] = []

        class MyCall(AudioCall):
            def __post_init__(self) -> None:
                super().__post_init__()
                created.append(self.caller)

        loop = asyncio.get_running_loop()
        protocol = self._CapturingSIP()
        protocol.transport = make_mock_transport()
        protocol.public_address = loop.create_future()
        protocol.public_address.set_result(("127.0.0.1", 5060))
        mux = RealtimeTransportProtocol()
        mux.public_address = loop.create_future()
        mux.public_address.set_result(("127.0.0.1", 12000))
        mock_rtp_transport = MagicMock()
        mock_rtp_transport.get_extra_info.return_value = ("127.0.0.1", 12000)
        protocol._rtp_protocol = mux
        protocol._rtp_transport = mock_rtp_transport
        request = make_invite()
        protocol._request_addrs[request.headers["Call-ID"]] = ("192.0.2.1", 5060)
        await protocol._answer(request, MyCall)
        assert created == ["sip:bob@biloxi.com"]

    async def test_answer__rtp_receives_audio(self):
        """Deliver audio from RTP packets to the call's audio_received via the RTP socket."""
        received_payloads: list[bytes] = []

        class AudioCapture(AudioCall):
            def _decode_raw(self, raw_packets: list[bytes]):
                # Skip real av decode; treat first payload as-is
                return raw_packets[0] if raw_packets else b""

            def audio_received(self, audio) -> None:
                received_payloads.append(audio)

        loop = asyncio.get_running_loop()
        protocol = self._CapturingSIP()
        protocol.transport = make_mock_transport()
        protocol.public_address = loop.create_future()
        protocol.public_address.set_result(("127.0.0.1", 5060))
        mux = RealtimeTransportProtocol(stun_server_address=("127.0.0.1", 65535))
        rtp_transport, _ = await loop.create_datagram_endpoint(
            lambda: mux, local_addr=("127.0.0.1", 0)
        )
        mux.public_address.set_result(rtp_transport.get_extra_info("sockname"))
        protocol._rtp_protocol = mux
        protocol._rtp_transport = rtp_transport
        request = make_invite()
        protocol._request_addrs[request.headers["Call-ID"]] = ("192.0.2.1", 5060)
        try:
            await protocol._answer(request, AudioCapture)
            response, _ = protocol._sent[0]
            sdp_line = next(
                line
                for line in bytes(response.body).decode().splitlines()
                if line.startswith("m=audio")
            )
            rtp_port = int(sdp_line.split()[1])

            send_transport, _ = await loop.create_datagram_endpoint(
                asyncio.DatagramProtocol,
                remote_addr=("127.0.0.1", rtp_port),
            )
            rtp_packet = b"\x80\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00audio"
            send_transport.sendto(rtp_packet)
            await asyncio.sleep(0.05)
            send_transport.close()
            assert received_payloads == [b"audio"]
        finally:
            rtp_transport.close()

    async def test_answer__rtp_receives_multiple_packets(self):
        """Call audio_received with each RTP payload when multiple packets arrive."""
        received_payloads: list[bytes] = []

        class AudioCapture(AudioCall):
            def _decode_raw(self, raw_packets: list[bytes]):
                return raw_packets[0] if raw_packets else b""

            def audio_received(self, audio) -> None:
                received_payloads.append(audio)

        loop = asyncio.get_running_loop()
        protocol = self._CapturingSIP()
        protocol.transport = make_mock_transport()
        protocol.public_address = loop.create_future()
        protocol.public_address.set_result(("127.0.0.1", 5060))
        mux = RealtimeTransportProtocol(stun_server_address=("127.0.0.1", 65535))
        rtp_transport, _ = await loop.create_datagram_endpoint(
            lambda: mux, local_addr=("127.0.0.1", 0)
        )
        mux.public_address.set_result(rtp_transport.get_extra_info("sockname"))
        protocol._rtp_protocol = mux
        protocol._rtp_transport = rtp_transport
        request = make_invite()
        protocol._request_addrs[request.headers["Call-ID"]] = ("192.0.2.1", 5060)
        try:
            await protocol._answer(request, AudioCapture)
            response, _ = protocol._sent[0]
            sdp_line = next(
                line
                for line in bytes(response.body).decode().splitlines()
                if line.startswith("m=audio")
            )
            rtp_port = int(sdp_line.split()[1])

            send_transport, _ = await loop.create_datagram_endpoint(
                asyncio.DatagramProtocol,
                remote_addr=("127.0.0.1", rtp_port),
            )
            header = b"\x80\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00"
            send_transport.sendto(header + b"chunk1")
            send_transport.sendto(header + b"chunk2")
            await asyncio.sleep(0.05)
            send_transport.close()
            assert received_payloads == [b"chunk1", b"chunk2"]
        finally:
            rtp_transport.close()

    async def test_answer__content_length_serialized(self):
        """Content-Length is automatically included when the response is serialized."""
        loop = asyncio.get_running_loop()
        protocol = self._CapturingSIP()
        protocol.transport = make_mock_transport()
        protocol.public_address = loop.create_future()
        protocol.public_address.set_result(("127.0.0.1", 5060))
        mux = RealtimeTransportProtocol()
        mux.public_address = loop.create_future()
        mux.public_address.set_result(("127.0.0.1", 12000))
        mock_rtp_transport = MagicMock()
        mock_rtp_transport.get_extra_info.return_value = ("127.0.0.1", 12000)
        protocol._rtp_protocol = mux
        protocol._rtp_transport = mock_rtp_transport
        request = make_invite()
        protocol._request_addrs[request.headers["Call-ID"]] = ("192.0.2.1", 5060)
        await protocol._answer(request, AudioCall)
        response, _ = protocol._sent[0]
        serialized = bytes(response)
        parsed = Message.parse(serialized)
        assert "Content-Length" in parsed.headers

    async def test_answer__reuses_shared_rtp_socket_for_second_call(self):
        """A second _answer() reuses the same shared RTP socket (one port for all calls)."""
        from voip.sdp.messages import SessionDescription  # noqa: PLC0415
        from voip.sip.messages import Request as SIPRequest  # noqa: PLC0415

        loop = asyncio.get_running_loop()
        protocol = self._CapturingSIP()
        protocol.transport = make_mock_transport()
        protocol.public_address = loop.create_future()
        protocol.public_address.set_result(("127.0.0.1", 5060))
        mux = RealtimeTransportProtocol(stun_server_address=("127.0.0.1", 65535))
        rtp_transport, _ = await loop.create_datagram_endpoint(
            lambda: mux, local_addr=("127.0.0.1", 0)
        )
        mux.public_address.set_result(rtp_transport.get_extra_info("sockname"))
        protocol._rtp_protocol = mux
        protocol._rtp_transport = rtp_transport

        sdp_body1 = SessionDescription.parse(
            b"v=0\r\no=- 0 0 IN IP4 1.2.3.4\r\ns=-\r\nc=IN IP4 1.2.3.4\r\nt=0 0\r\nm=audio 5000 RTP/AVP 0\r\n"
        )
        invite1 = SIPRequest(
            method="INVITE",
            uri="sip:alice@atlanta.com",
            headers={
                "Via": "SIP/2.0/UDP pc33.atlanta.com",
                "To": "sip:alice@atlanta.com",
                "From": "sip:bob@biloxi.com",
                "Call-ID": "call-1@test",
                "CSeq": "1 INVITE",
            },
            body=sdp_body1,
        )
        protocol._request_addrs["call-1@test"] = ("1.2.3.4", 5060)
        await protocol._answer(invite1, AudioCall)
        rtp_proto_1 = protocol._rtp_protocol
        rtp_transport_1 = protocol._rtp_transport

        sdp_body2 = SessionDescription.parse(
            b"v=0\r\no=- 0 0 IN IP4 5.6.7.8\r\ns=-\r\nc=IN IP4 5.6.7.8\r\nt=0 0\r\nm=audio 6000 RTP/AVP 0\r\n"
        )
        invite2 = SIPRequest(
            method="INVITE",
            uri="sip:alice@atlanta.com",
            headers={
                "Via": "SIP/2.0/UDP pc33.atlanta.com",
                "To": "sip:alice@atlanta.com",
                "From": "sip:charlie@biloxi.com",
                "Call-ID": "call-2@test",
                "CSeq": "1 INVITE",
            },
            body=sdp_body2,
        )
        protocol._request_addrs["call-2@test"] = ("5.6.7.8", 5060)
        await protocol._answer(invite2, AudioCall)

        # The same RTP protocol and transport must be reused.
        assert protocol._rtp_protocol is rtp_proto_1
        assert protocol._rtp_transport is rtp_transport_1

        # Both calls registered under their respective remote addrs.
        assert ("1.2.3.4", 5000) in rtp_proto_1.calls
        assert ("5.6.7.8", 6000) in rtp_proto_1.calls

        # Clean up the real socket.
        rtp_transport.close()

    async def test_answer__bye_unregisters_call_from_rtp_mux(self):
        """BYE for an active call removes its handler from the shared RTP mux."""
        loop = asyncio.get_running_loop()
        protocol = self._CapturingSIP()
        protocol.transport = make_mock_transport()
        protocol.public_address = loop.create_future()
        protocol.public_address.set_result(("127.0.0.1", 5060))
        mux = RealtimeTransportProtocol(stun_server_address=("127.0.0.1", 65535))
        rtp_transport, _ = await loop.create_datagram_endpoint(
            lambda: mux, local_addr=("127.0.0.1", 0)
        )
        mux.public_address.set_result(rtp_transport.get_extra_info("sockname"))
        protocol._rtp_protocol = mux
        protocol._rtp_transport = rtp_transport
        request = make_invite()
        protocol._request_addrs[request.headers["Call-ID"]] = ("192.0.2.1", 5060)
        try:
            await protocol._answer(request, AudioCall)

            # The call is registered under the None wildcard (no SDP in invite).
            assert None in mux.calls

            bye = Request(
                method="BYE",
                uri="sip:alice@atlanta.com",
                headers={
                    "Via": "SIP/2.0/UDP pc33.atlanta.com",
                    "To": "sip:alice@atlanta.com",
                    "From": "sip:bob@biloxi.com",
                    "Call-ID": request.headers["Call-ID"],
                    "CSeq": "2 BYE",
                },
            )
            protocol.request_received(bye, ("192.0.2.1", 5060))
            assert None not in mux.calls
        finally:
            rtp_transport.close()

    async def test_answer__logs_info(self, caplog):
        """Log an info message when answering a call."""
        import logging

        loop = asyncio.get_running_loop()
        protocol = self._CapturingSIP()
        protocol.transport = make_mock_transport()
        protocol.public_address = loop.create_future()
        protocol.public_address.set_result(("127.0.0.1", 5060))
        mux = RealtimeTransportProtocol()
        mux.public_address = loop.create_future()
        mux.public_address.set_result(("127.0.0.1", 12000))
        mock_rtp_transport = MagicMock()
        mock_rtp_transport.get_extra_info.return_value = ("127.0.0.1", 12000)
        protocol._rtp_protocol = mux
        protocol._rtp_transport = mock_rtp_transport
        request = make_invite()
        protocol._request_addrs[request.headers["Call-ID"]] = ("192.0.2.1", 5060)
        with caplog.at_level(logging.INFO, logger="voip.sip"):
            await protocol._answer(request, AudioCall)
        assert any("call_answered" in r.message for r in caplog.records)

    def test_reject__sends_busy_here_by_default(self):
        """Send a 486 Busy Here response when no status code is given."""
        protocol = self._CapturingSIP()
        request = make_invite()
        protocol._request_addrs[request.headers["Call-ID"]] = ("192.0.2.1", 5060)
        protocol.reject(request)
        assert len(protocol._sent) == 1
        response, addr = protocol._sent[0]
        assert isinstance(response, Response)
        assert response.status_code == 486
        assert response.reason == "Busy Here"
        assert addr == ("192.0.2.1", 5060)

    def test_reject__custom_status(self):
        """Send the specified status code and reason."""
        protocol = self._CapturingSIP()
        request = make_invite()
        protocol._request_addrs[request.headers["Call-ID"]] = ("192.0.2.1", 5060)
        protocol.reject(request, status_code=603, reason="Decline")
        response, _ = protocol._sent[0]
        assert response.status_code == 603
        assert response.reason == "Decline"

    def test_reject__copies_dialog_headers(self):
        """Copy Via, To, From, Call-ID, and CSeq headers into the response."""
        protocol = self._CapturingSIP()
        request = make_invite()
        protocol._request_addrs[request.headers["Call-ID"]] = ("192.0.2.1", 5060)
        protocol.reject(request)
        response, _ = protocol._sent[0]
        assert response.headers["Via"] == "SIP/2.0/UDP pc33.atlanta.com"
        assert response.headers["To"] == "sip:alice@atlanta.com"
        assert response.headers["From"] == "sip:bob@biloxi.com"
        assert response.headers["Call-ID"] == "1234@pc33"
        assert response.headers["CSeq"] == "1 INVITE"

    @pytest.mark.parametrize("extra_header", ["X-Custom"])
    def test_reject__excludes_extra_headers(self, extra_header):
        """Exclude non-dialog headers from the reject response."""
        protocol = self._CapturingSIP()
        request = make_invite({extra_header: "value"})
        protocol._request_addrs[request.headers["Call-ID"]] = ("192.0.2.1", 5060)
        protocol.reject(request)
        response, _ = protocol._sent[0]
        assert extra_header not in response.headers

    def test_reject__logs_info(self, caplog):
        """Log an info message when rejecting a call."""
        import logging

        with caplog.at_level(logging.INFO, logger="voip.sip"):
            protocol = self._CapturingSIP()
            request = make_invite()
            protocol._request_addrs[request.headers["Call-ID"]] = ("192.0.2.1", 5060)
            protocol.reject(request)
        assert any("call_rejected" in r.message for r in caplog.records)

    async def test_datagram_received__keepalive__sends_pong(self):
        """Double-CRLF keepalive (RFC 5626 §4.4.1) is answered with a single-CRLF pong."""
        protocol = SIP(server_address=("127.0.0.1", 5060), aor="sip:test@example.com")
        transport = MagicMock()
        protocol.connection_made(transport)
        transport.sendto.reset_mock()  # clear the STUN request call
        addr = ("192.0.2.1", 5060)
        protocol.datagram_received(b"\r\n\r\n", addr)
        transport.sendto.assert_called_once_with(b"\r\n", addr)

    async def test_request_received__unsupported_method__raises(self):
        """Raise NotImplementedError for any non-INVITE SIP request method."""
        protocol = SIP(server_address=("127.0.0.1", 5060), aor="sip:test@example.com")
        protocol.connection_made(MagicMock())
        request = Request(method="OPTIONS", uri="sip:alice@atlanta.com")
        with pytest.raises(NotImplementedError, match="OPTIONS"):
            protocol.request_received(request, ("192.0.2.1", 5060))

    async def test_answer__via_call_received__schedules_answer(self):
        """answer() is async; wrapping it in create_task from call_received works."""
        answered = []

        class MySIP(SIP):
            def call_received(self, request):
                asyncio.create_task(self.answer(request=request, call_class=AudioCall))

            async def _answer(self, request, call_class):
                answered.append((request, call_class))

        protocol = MySIP(server_address=("127.0.0.1", 5060), aor="sip:test@example.com")
        protocol.connection_made(MagicMock())
        request = make_invite()
        addr = ("192.0.2.1", 5060)
        protocol._request_addrs[request.headers["Call-ID"]] = addr
        protocol.call_received(request)

        await asyncio.sleep(0.01)
        assert len(answered) == 1
        assert answered[0][1] is AudioCall


class TestSessionInitiationProtocol:
    def test_registrar_uri__strips_user_from_aor(self):
        """Derive registrar URI from AOR by stripping the user part."""
        p = make_register_session(aor="sip:alice@example.com")
        assert p.registrar_uri == "sip:example.com"

    def test_registrar_uri__preserves_port(self):
        """Preserve a non-default port in the derived registrar URI."""
        p = make_register_session(aor="sip:alice@example.com:5080")
        assert p.registrar_uri == "sip:example.com:5080"

    async def test_connection_made__sends_register(self):
        """Send a REGISTER request immediately when connection is established."""

        class _SessionNoRTP(SessionInitiationProtocol):
            async def _start_rtp_mux(self):
                pass  # Avoid real socket + STUN in unit test

        p = _SessionNoRTP(
            server_address=("192.0.2.2", 5060),
            aor="sip:alice@example.com",
            username="alice",
            password="secret",  # noqa: S106
        )
        transport = make_mock_transport()
        p.connection_made(transport)
        p.public_address.set_result(("127.0.0.1", 5060))
        await asyncio.sleep(0.05)
        transport.sendto.assert_called()
        data, addr = transport.sendto.call_args[0]
        assert b"REGISTER sip:example.com SIP/2.0" in data
        assert addr == ("192.0.2.2", 5060)

    async def test_register__includes_required_headers(self):
        """REGISTER request includes From, To, Call-ID, CSeq, Contact and Expires."""
        loop = asyncio.get_running_loop()
        p = make_register_session()
        transport = make_mock_transport()
        p.transport = transport
        p.public_address = loop.create_future()
        p.public_address.set_result(("127.0.0.1", 5060))
        await p.register()
        data, _ = transport.sendto.call_args[0]
        assert b"From: sip:alice@example.com" in data
        assert b"To: sip:alice@example.com" in data
        assert b"Contact: <sip:alice@127.0.0.1:5060>" in data
        assert b"Expires: 3600" in data

    async def test_register__increments_cseq(self):
        """CSeq increments with each REGISTER sent."""
        loop = asyncio.get_running_loop()
        p = make_register_session()
        p.public_address = loop.create_future()
        p.public_address.set_result(("127.0.0.1", 5060))
        p.transport = make_mock_transport()
        await p.register()
        assert p.cseq == 1
        await p.register()
        assert p.cseq == 2

    async def test_register__with_authorization(self):
        """Authorization header is included when credentials are provided."""
        loop = asyncio.get_running_loop()
        p = make_register_session()
        transport = make_mock_transport()
        p.transport = transport
        p.public_address = loop.create_future()
        p.public_address.set_result(("127.0.0.1", 5060))
        await p.register(authorization='Digest username="alice"')
        data, _ = transport.sendto.call_args[0]
        assert b'Authorization: Digest username="alice"' in data

    async def test_register__with_proxy_authorization(self):
        """Proxy-Authorization header is included for proxy challenges."""
        loop = asyncio.get_running_loop()
        p = make_register_session()
        transport = make_mock_transport()
        p.transport = transport
        p.public_address = loop.create_future()
        p.public_address.set_result(("127.0.0.1", 5060))
        await p.register(proxy_authorization='Digest username="alice"')
        data, _ = transport.sendto.call_args[0]
        assert b'Proxy-Authorization: Digest username="alice"' in data

    async def test_response_received__200_ok_calls_registered(self):
        """Receiving 200 OK for REGISTER triggers registered()."""
        calls = []

        class ConcreteSession(SessionInitiationProtocol):
            def registered(self):
                calls.append(True)

        p = ConcreteSession(
            server_address=("192.0.2.2", 5060),
            aor="sip:alice@example.com",
            username="a",
            password="b",  # noqa: S106
        )
        p.connection_made(make_mock_transport())
        p.response_received(
            Response(status_code=200, reason="OK", headers={"CSeq": "1 REGISTER"}),
            ("192.0.2.2", 5060),
        )
        assert calls == [True]

    async def test_response_received__200_non_register_raises(self):
        """Receiving 200 OK for a non-REGISTER method raises NotImplementedError."""
        p = make_register_session()
        p.connection_made(make_mock_transport())
        with pytest.raises(NotImplementedError):
            p.response_received(
                Response(status_code=200, reason="OK", headers={"CSeq": "1 INVITE"}),
                ("192.0.2.2", 5060),
            )

    async def test_response_received__401_retries_with_authorization(self):
        """Receiving 401 triggers a re-REGISTER with an Authorization header."""
        loop = asyncio.get_running_loop()
        p = make_register_session(username="alice", password="secret")  # noqa: S106
        transport = make_mock_transport()
        p.transport = transport
        p.public_address = loop.create_future()
        p.public_address.set_result(("127.0.0.1", 5060))
        challenge = 'Digest realm="example.com", nonce="abc123"'
        p.response_received(
            Response(
                status_code=401,
                reason="Unauthorized",
                headers={"WWW-Authenticate": challenge, "CSeq": "1 REGISTER"},
            ),
            ("192.0.2.2", 5060),
        )
        await asyncio.sleep(0.05)
        data, _ = transport.sendto.call_args[0]
        assert b"Authorization: Digest" in data
        assert b'username="alice"' in data
        assert b'realm="example.com"' in data
        assert b'nonce="abc123"' in data
        assert b'algorithm="MD5"' in data

    async def test_response_received__407_retries_with_proxy_authorization(self):
        """Receiving 407 triggers a re-REGISTER with a Proxy-Authorization header."""
        loop = asyncio.get_running_loop()
        p = make_register_session(username="alice", password="secret")  # noqa: S106
        transport = make_mock_transport()
        p.transport = transport
        p.public_address = loop.create_future()
        p.public_address.set_result(("127.0.0.1", 5060))
        challenge = 'Digest realm="example.com", nonce="xyz"'
        p.response_received(
            Response(
                status_code=407,
                reason="Proxy Auth Required",
                headers={"Proxy-Authenticate": challenge, "CSeq": "1 REGISTER"},
            ),
            ("192.0.2.2", 5060),
        )
        await asyncio.sleep(0.05)
        data, _ = transport.sendto.call_args[0]
        assert b"Proxy-Authorization: Digest" in data
        assert b'username="alice"' in data

    async def test_response_received__401_with_qop_auth_includes_nc_cnonce(self):
        """401 with qop=auth causes the retry to include nc and cnonce fields."""
        loop = asyncio.get_running_loop()
        p = make_register_session()
        transport = make_mock_transport()
        p.transport = transport
        p.public_address = loop.create_future()
        p.public_address.set_result(("127.0.0.1", 5060))
        challenge = 'Digest realm="example.com", nonce="n", qop="auth"'
        p.response_received(
            Response(
                status_code=401,
                reason="Unauthorized",
                headers={"WWW-Authenticate": challenge, "CSeq": "1 REGISTER"},
            ),
            ("192.0.2.2", 5060),
        )
        await asyncio.sleep(0.05)
        data, _ = transport.sendto.call_args[0]
        assert b"qop=auth" in data
        assert b"nc=00000001" in data
        assert b"cnonce=" in data

    async def test_response_received__401_with_opaque_echoes_opaque(self):
        """The opaque field from the challenge is echoed back in the Authorization."""
        loop = asyncio.get_running_loop()
        p = make_register_session()
        transport = make_mock_transport()
        p.transport = transport
        p.public_address = loop.create_future()
        p.public_address.set_result(("127.0.0.1", 5060))
        challenge = 'Digest realm="example.com", nonce="n", opaque="secret-opaque"'
        p.response_received(
            Response(
                status_code=401,
                reason="Unauthorized",
                headers={"WWW-Authenticate": challenge, "CSeq": "1 REGISTER"},
            ),
            ("192.0.2.2", 5060),
        )
        await asyncio.sleep(0.05)
        data, _ = transport.sendto.call_args[0]
        assert b'opaque="secret-opaque"' in data

    async def test_register__via_header_has_rport(self):
        """REGISTER request includes a Via header with the rport parameter for NAT traversal."""
        import re

        loop = asyncio.get_running_loop()
        p = make_register_session()
        p.public_address = loop.create_future()
        p.public_address.set_result(("192.0.2.10", 5060))
        transport = make_mock_transport("192.0.2.10", 5060)
        p.transport = transport
        await p.register()
        data, _ = transport.sendto.call_args[0]
        assert b"Via: SIP/2.0/UDP 192.0.2.10:5060;rport;branch=z9hG4bK" in data
        assert re.search(rb"branch=z9hG4bK[0-9a-f]{32}", data)

    async def test_register__via_branch_is_unique_per_request(self):
        """Each REGISTER generates a unique Via branch to prevent transaction conflicts."""
        import re

        loop = asyncio.get_running_loop()
        p = make_register_session()
        p.public_address = loop.create_future()
        p.public_address.set_result(("127.0.0.1", 5060))
        transport = make_mock_transport()
        p.transport = transport
        await p.register()
        data1, _ = transport.sendto.call_args[0]
        transport.reset_mock()
        await p.register()
        data2, _ = transport.sendto.call_args[0]
        branch1 = re.search(rb"branch=(z9hG4bK[0-9a-f]{32})", data1).group(1)
        branch2 = re.search(rb"branch=(z9hG4bK[0-9a-f]{32})", data2).group(1)
        assert branch1 != branch2

    async def test_register__contact_uses_local_addr(self):
        """Contact header always uses the local socket address."""
        loop = asyncio.get_running_loop()
        p = make_register_session()
        p.public_address = loop.create_future()
        p.public_address.set_result(("10.0.0.5", 5060))
        transport = make_mock_transport("10.0.0.5", 5060)
        p.transport = transport
        await p.register()
        data, _ = transport.sendto.call_args[0]
        assert b"Contact: <sip:alice@10.0.0.5:5060>" in data

    async def test_datagram_received__sip_response__calls_response_received(self):
        """datagram_received routes SIP messages to response_received."""
        received = []

        class ConcreteSession(SessionInitiationProtocol):
            def response_received(self, response, addr):
                received.append(response)

        p = ConcreteSession(
            server_address=("192.0.2.2", 5060),
            aor="sip:alice@example.com",
            username="a",
            password="b",  # noqa: S106
        )
        p.connection_made(make_mock_transport())
        sip_data = b"SIP/2.0 200 OK\r\nCSeq: 1 REGISTER\r\n\r\n"
        p.datagram_received(sip_data, ("192.0.2.2", 5060))
        assert len(received) == 1
        assert received[0].status_code == 200

    async def test_invite_received_after_register(self):
        """INVITE dispatching still works after registration (call_received is called)."""
        received = []

        class ConcreteSession(SessionInitiationProtocol):
            def call_received(self, request):
                received.append(request)

        p = ConcreteSession(
            server_address=("192.0.2.2", 5060),
            aor="sip:alice@example.com",
            username="a",
            password="b",  # noqa: S106
        )
        p.connection_made(make_mock_transport())
        request = Request(
            method="INVITE",
            uri="sip:alice@example.com",
            headers={"From": "sip:bob@example.com", "Call-ID": "test@pc"},
        )
        p.request_received(request, ("192.0.2.1", 5060))
        assert len(received) == 1
        assert received[0] is request

    async def test_response_received__200_ok__logs_info(self, caplog):
        """Receiving 200 OK logs an info message."""
        import logging

        p = make_register_session()
        p.connection_made(make_mock_transport())
        with caplog.at_level(logging.INFO, logger="voip.sip"):
            p.response_received(
                Response(status_code=200, reason="OK", headers={"CSeq": "1 REGISTER"}),
                ("192.0.2.2", 5060),
            )
        assert any("Registration successful" in r.message for r in caplog.records)

    async def test_response_received__unexpected_status__logs_warning(self, caplog):
        """An unhandled status code logs a warning and raises NotImplementedError."""
        import logging

        p = make_register_session()
        p.connection_made(make_mock_transport())
        with caplog.at_level(logging.WARNING, logger="voip.sip"):
            with pytest.raises(NotImplementedError):
                p.response_received(
                    Response(
                        status_code=500,
                        reason="Server Error",
                        headers={"CSeq": "1 REGISTER"},
                    ),
                    ("192.0.2.2", 5060),
                )
        assert any("500" in r.message for r in caplog.records)


def make_invite(headers: dict | None = None) -> Request:
    """Return an INVITE request with default headers."""
    return Request(
        method="INVITE",
        uri="sip:alice@atlanta.com",
        headers={
            "Via": "SIP/2.0/UDP pc33.atlanta.com",
            "To": "sip:alice@atlanta.com",
            "From": "sip:bob@biloxi.com",
            "Call-ID": "1234@pc33",
            "CSeq": "1 INVITE",
            **(headers or {}),
        },
    )


def make_register_session(
    server_addr=("192.0.2.2", 5060),
    aor="sip:alice@example.com",
    username="alice",
    password="secret",  # noqa: S107
) -> SessionInitiationProtocol:
    """Return a SessionInitiationProtocol session without triggering connection_made."""
    return SessionInitiationProtocol(
        server_address=server_addr,
        aor=aor,
        username=username,
        password=password,
    )


def make_mock_transport(host: str = "127.0.0.1", port: int = 5060):
    """Return a MagicMock transport with get_extra_info('sockname') configured."""
    transport = MagicMock()
    transport.get_extra_info.return_value = (host, port)
    return transport
