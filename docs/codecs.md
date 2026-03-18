# Codecs

## Overview

VoIP ships two tiers of audio codecs:

| Extra required                           | Codecs available                       |
| ---------------------------------------- | -------------------------------------- |
| `audio` (includes [numpy])               | PCMA (G.711 A-law), PCMU (G.711 µ-law) |
| `hd-audio` (includes [numpy] and [pyav]) | + G.722, Opus                          |

Install the minimal tier for pure-Python telephony deployments:

```bash
pip install voip[audio]
```

Install the full tier for wideband / Opus support via [FFmpeg]:

```bash
pip install voip[hd-audio]
```

## SD audio

These codecs work without PyAV and require only `numpy`.

::: voip.codecs.pcma.PCMA

::: voip.codecs.pcmu.PCMU

## HD audio

These codecs require the `hd-audio` extra (`pip install voip[hd-audio]`).

::: voip.codecs.g722.G722

::: voip.codecs.g722.G722Decoder

::: voip.codecs.opus.Opus

## Registry

::: voip.codecs.get

## Base classes

::: voip.codecs.base.RTPCodec

::: voip.codecs.base.PayloadDecoder

::: voip.codecs.base.PerPacketDecoder

::: voip.codecs.av.PyAVCodec

[ffmpeg]: https://ffmpeg.org/
[numpy]: https://numpy.org/
[pyav]: https://pyav.org/
