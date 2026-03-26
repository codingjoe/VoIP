class RegistrationError(Exception):
    """Raised when a SIP REGISTER request fails with an unexpected response.

    The exception message includes the response status code and reason phrase
    from the server, e.g. ``"403 Forbidden"`` or ``"500 Server Error"``.
    """
