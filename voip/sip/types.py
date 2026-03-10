from __future__ import annotations

import enum

Status = enum.IntEnum(
    value="Status",
    names={
        "Ringing": 180,
        "OK": 200,
        "Unauthorized": 401,
        "Proxy Authentication Required": 407,
        "Busy Here": 486,
        "Request Terminated": 487,
    },
)
