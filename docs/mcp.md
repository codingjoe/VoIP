# MCP Server

The `voip` package ships a ready-made [Model Context Protocol (MCP)][MCP] server that
exposes `say` and `call` tools so that any MCP client (e.g. Claude Code) can make phone
calls on your behalf.

[MCP]: https://modelcontextprotocol.io/

## Claude Code setup

Add the server to your MCP config (see [Claude Code MCP docs][cc-mcp]):

```json
{
  "mcpServers": {
    "voip": {
      "command": "voip",
      "args": ["mcp"],
      "env": {
        "SIP_AOR": "sip:youruser@carrier.example"
      }
    }
  }
}
```

Set `SIP_AOR` to your SIP address-of-record.  The transport (TLS vs TCP) and proxy
address are derived from the URI automatically.

[cc-mcp]: https://docs.anthropic.com/en/docs/claude-code/mcp

## Tools

::: voip.mcp.say

::: voip.mcp.call

## Session lifecycle

::: voip.mcp.run

::: voip.mcp.HangupDialog

::: voip.mcp.MCPAgentCall
