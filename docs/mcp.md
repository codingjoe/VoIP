# MCP Server

The `voip` package ships a ready-made [Model Context Protocol (MCP)][mcp] server
that exposes tools to make phone calls on your behalf to any MCP client.

## Claude Code setup

Add the server to your MCP config (see [Claude Code MCP docs][cc-mcp]):

```json
{
  "mcpServers": {
    "VoIP": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "voip[mcp]",
        "mcp"
      ],
      "env": {
        "SIP_AOR": "sip:****:****@example.com:5060?transport=tcp"
      }
    }
  }
}
```

Set `SIP_AOR` to your SIP address-of-record.

## Tools

::: voip.mcp.say

::: voip.mcp.send_fax

::: voip.mcp.call

[cc-mcp]: https://docs.anthropic.com/en/docs/claude-code/mcp
[mcp]: https://modelcontextprotocol.io/
