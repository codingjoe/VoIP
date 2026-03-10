from __future__ import annotations

import enum

Status = enum.IntEnum(
    value="Status",
    names={
        # 1xx Provisional
        "Trying": 100,
        "Ringing": 180,
        "Call Is Being Forwarded": 181,
        "Queued": 182,
        "Session Progress": 183,
        # 2xx Success
        "OK": 200,
        # 3xx Redirection
        "Multiple Choices": 300,
        "Moved Permanently": 301,
        "Moved Temporarily": 302,
        "Use Proxy": 305,
        "Alternative Service": 380,
        # 4xx Client Failure
        "Bad Request": 400,
        "Unauthorized": 401,
        "Payment Required": 402,
        "Forbidden": 403,
        "Not Found": 404,
        "Method Not Allowed": 405,
        "Not Acceptable": 406,
        "Proxy Authentication Required": 407,
        "Request Timeout": 408,
        "Gone": 410,
        "Request Entity Too Large": 413,
        "Request-URI Too Long": 414,
        "Unsupported Media Type": 415,
        "Unsupported URI Scheme": 416,
        "Bad Extension": 420,
        "Extension Required": 421,
        "Interval Too Brief": 423,
        "Temporarily Unavailable": 480,
        "Call/Transaction Does Not Exist": 481,
        "Loop Detected": 482,
        "Too Many Hops": 483,
        "Address Incomplete": 484,
        "Ambiguous": 485,
        "Busy Here": 486,
        "Request Terminated": 487,
        "Not Acceptable Here": 488,
        "Request Pending": 491,
        "Undecipherable": 493,
        # 5xx Server Failure
        "Server Internal Error": 500,
        "Not Implemented": 501,
        "Bad Gateway": 502,
        "Service Unavailable": 503,
        "Server Time-out": 504,
        "Version Not Supported": 505,
        "Message Too Large": 513,
        # 6xx Global Failure
        "Busy Everywhere": 600,
        "Decline": 603,
        "Does Not Exist Anywhere": 604,
    },
)
