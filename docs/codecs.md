# Codecs

## Overview

VoIP ships two tiers of audio codecs:

| Extra required            | Codecs available                       |
| ------------------------- | -------------------------------------- |
| `numpy`                   | PCMA (G.711 A-law), PCMU (G.711 µ-law) |
| `pyav` (includes `numpy`) | + G.722, Opus                          |

Install the minimal tier for pure-Python telephony deployments:

```bash
pip install voip[audio]
```

Install the full tier for wideband / Opus support via [FFmpeg]:

```bash
pip install voip[hd-audio]
```

## Base classes

::: voip.codecs.base

::: voip.codecs.av

## Pure-NumPy codecs

These codecs work without PyAV and require only `numpy`.

::: voip.codecs.pcma

::: voip.codecs.pcmu

## PyAV codecs

These codecs require the `pyav` extra (`pip install voip[pyav]`).

::: voip.codecs.g722

::: voip.codecs.opus

## Registry

::: voip.codecs.get

[ffmpeg]: https://ffmpeg.org/
