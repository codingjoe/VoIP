"""PyAV-based codec base class for RTP audio streams.

All concrete codec classes in this package inherit from [`PyAVCodec`][voip.codecs.av.PyAVCodec],
which provides shared [`decode_pcm`][voip.codecs.av.PyAVCodec.decode_pcm] and
[`encode_pcm`][voip.codecs.av.PyAVCodec.encode_pcm] helpers backed by [PyAV][].

[PyAV]: https://pyav.basswood-io.com/
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from typing import TYPE_CHECKING, ClassVar, cast

import av
import av.audio.resampler
import numpy as np

from voip.sdp.types import RTPPayloadFormat

if TYPE_CHECKING:
    pass

__all__ = ["RTPCodec"]


class RTPCodec:
    """Base class for RTP audio codecs that decode and encode via PyAV.

    Concrete subclasses define codec-specific class variables and may override
    [`decode`][voip.codecs.av.PyAVCodec.decode],
    [`encode`][voip.codecs.av.PyAVCodec.encode], and
    [`packetize`][voip.codecs.av.PyAVCodec.packetize].

    Subclasses that produce variable-length output across frames (e.g. G.722
    ADPCM) should override `packetize` to encode the whole buffer at once and
    preserve predictor state.
    """

    payload_type: ClassVar[int]
    """RTP payload type number."""

    encoding_name: ClassVar[str]
    """SDP encoding name (lowercase)."""

    sample_rate_hz: ClassVar[int]
    """Actual audio sample rate in Hz."""

    rtp_clock_rate_hz: ClassVar[int]
    """RTP timestamp clock rate in Hz (may differ from `sample_rate_hz`)."""

    frame_size: ClassVar[int]
    """Audio samples per 20 ms RTP frame at `sample_rate_hz`."""

    timestamp_increment: ClassVar[int]
    """RTP timestamp ticks per frame at `rtp_clock_rate_hz`."""

    channels: ClassVar[int]
    """Channel count (1 = mono, 2 = stereo)."""

    @classmethod
    def decode_pcm(
        cls,
        data: bytes,
        av_format: str,
        output_rate_hz: int,
        *,
        input_rate_hz: int | None = None,
    ) -> np.ndarray:
        """Decode raw audio bytes via PyAV into float32 mono PCM.

        Args:
            data: Raw audio bytes in the codec's wire format.
            av_format: PyAV format string (e.g. `"ogg"`, `"alaw"`).
            output_rate_hz: Target sample rate in Hz.
            input_rate_hz: Input clock rate hint for the PyAV decoder, or
                `None` for self-describing formats like Ogg.

        Returns:
            Float32 mono PCM array at *output_rate_hz* Hz.
        """
        resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout="mono", rate=output_rate_hz
        )
        frames: list[np.ndarray] = []
        with av.open(
            io.BytesIO(data),
            mode="r",
            format=av_format,
            options=(
                {"sample_rate": str(input_rate_hz)} if input_rate_hz is not None else {}
            ),
        ) as container:
            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    frames.append(resampled.to_ndarray().flatten())
        for resampled in resampler.resample(None):
            frames.append(resampled.to_ndarray().flatten())
        return np.concatenate(frames) if frames else np.array([], dtype=np.float32)

    @classmethod
    def encode_pcm(
        cls,
        samples: np.ndarray,
        av_codec_name: str,
        sample_rate_hz: int,
    ) -> bytes:
        """Encode float32 mono PCM to raw codec bytes via PyAV.

        Args:
            samples: Float32 mono PCM array in the range `[-1, 1]`.
            av_codec_name: PyAV codec name (e.g. `"g722"` or `"libopus"`).
            sample_rate_hz: Sample rate of *samples* in Hz.

        Returns:
            Encoded audio bytes.
        """
        codec: av.AudioCodecContext = cast(
            av.AudioCodecContext, av.CodecContext.create(av_codec_name, "w")
        )
        codec.sample_rate = sample_rate_hz
        codec.format = av.AudioFormat("s16")
        codec.layout = av.AudioLayout("mono")
        codec.open()
        pcm = np.clip(np.round(samples * 32768.0), -32768, 32767).astype(np.int16)
        frame = av.AudioFrame.from_ndarray(
            pcm[np.newaxis, :], format="s16", layout="mono"
        )
        frame.sample_rate = sample_rate_hz
        frame.pts = 0
        return b"".join(
            bytes(packet)
            for segment in (codec.encode(frame), codec.encode(None))
            for packet in segment
        )

    @classmethod
    def to_payload_format(cls) -> RTPPayloadFormat:
        """Create an [`RTPPayloadFormat`][voip.sdp.types.RTPPayloadFormat] for SDP negotiation.

        Uses `rtp_clock_rate_hz` as the SDP sample rate, which is correct
        per RFC 3551 (e.g. G.722 advertises 8000 Hz in SDP even though the
        actual audio runs at 16000 Hz).

        Returns:
            Payload format descriptor for this codec.
        """
        return RTPPayloadFormat(
            payload_type=cls.payload_type,
            encoding_name=cls.encoding_name,
            sample_rate=cls.rtp_clock_rate_hz,
            channels=cls.channels,
        )

    @classmethod
    def decode(
        cls,
        payload: bytes,
        output_rate_hz: int,
        *,
        input_rate_hz: int | None = None,
    ) -> np.ndarray:
        """Decode an RTP payload to float32 mono PCM.

        Override in subclasses to wrap the payload in a container format
        (e.g. Ogg for Opus) or select a codec-specific PyAV format string.

        Args:
            payload: Raw RTP payload bytes.
            output_rate_hz: Target sample rate in Hz.
            input_rate_hz: Input clock rate override in Hz, or `None` to use
                the subclass default.

        Returns:
            Float32 mono PCM array at *output_rate_hz* Hz.
        """
        raise NotImplementedError(f"{cls.__name__} does not implement decode.")

    @classmethod
    def encode(cls, samples: np.ndarray) -> bytes:
        """Encode float32 mono PCM to an RTP payload.

        Override in subclasses to implement codec-specific encoding.

        Args:
            samples: Float32 mono PCM at `sample_rate_hz` Hz.

        Returns:
            Encoded bytes for one RTP payload.
        """
        raise NotImplementedError(f"{cls.__name__} does not implement encode.")

    @classmethod
    def packetize(cls, audio: np.ndarray) -> Iterator[bytes]:
        """Encode *audio* and yield one encoded payload per 20 ms RTP frame.

        The default implementation encodes one `frame_size` chunk at a time
        using `encode`. Override in subclasses (e.g. G.722) where the entire
        buffer must be encoded at once to preserve codec state.

        Args:
            audio: Float32 mono PCM at `sample_rate_hz` Hz.

        Yields:
            Encoded payload bytes, one per RTP packet.
        """
        for i in range(0, len(audio), cls.frame_size):
            yield cls.encode(audio[i : i + cls.frame_size])
