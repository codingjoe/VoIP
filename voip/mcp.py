import asyncio

from fastmcp import FastMCP

import voip

mcp = FastMCP(
    "VoIP",
    "Provide a set of tools to make phone calls and send text messages.",
    version=voip.__version__,
    website_url="https://codingjoe.dev/VoIP/",
)


@mcp.tool
def say(target: str, prompt="") -> None:
    """
    Call the phone number and say the prompt.

    Args:
        target: The phone number to say.
        prompt: The prompt to say.
    """


@mcp.tool
def call(target: str, initial_prompt="", system_prompt: str = "") -> str:
    """
    Call the phone number, have a conversation and return transcript.

    Args:
        target: The phone number to call as a tel-link including the country code, e.g. "tel:+1234567890".
        initial_prompt: The initial prompt for the call, e.g. "Hello, I am calling to ask about your product."
        system_prompt: The system prompt used by the LLM to guide the conversation, e.g. "You are a customer service representative for a company that sells products. You will answer the customer's questions and provide information about the products."
    """


if __name__ == "__main__":
    asyncio.run(mcp.run_async(transport="http"))
