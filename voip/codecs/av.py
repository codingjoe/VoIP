"""PyAV-backed RTP codec base class.

[`PyAVCodec`][voip.codecs.av.PyAVCodec] extends
[`RTPCodec`][voip.codecs.base.RTPCodec] with
[`decode_pcm`][voip.codecs.av.PyAVCodec.decode_pcm] and
[`encode_pcm`][voip.codecs.av.PyAVCodec.encode_pcm] helpers that use
[PyAV][] for container-aware decode and codec-aware encode.

Requires the ``pyav`` extra: ``pip install voip[pyav]``.

Concrete subclasses: [`Opus`][voip.codecs.Opus], [`G722`][voip.codecs.G722].

[PyAV]: https://pyav.basswood-io.com/
"""

from __future__ import annotations

import io
from typing import cast

import av
import av.audio.resampler
import numpy as np

from voip.codecs.base import RTPCodec

__all__ = ["PyAVCodec"]


class PyAVCodec(RTPCodec):
    """RTP codec that decodes and encodes audio via [PyAV][].

    Concrete implementations: [`Opus`][voip.codecs.Opus],
    [`G722`][voip.codecs.G722].

    [PyAV]: https://pyav.basswood-io.com/
    """

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
