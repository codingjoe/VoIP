# MCP Server & Skills

The `voip` package ships a ready-made [Model Context Protocol (MCP)][mcp] server
that exposes tools to make phone calls on your behalf to any MCP client.

For Claude Code users, the repository root also contains a [Claude Code plugin][plugins]
that bundles the MCP server and adds dedicated skills for each VoIP workflow
(see [Plugin](#claude-code-plugin) below).

## Claude & other agentic frameworks

### Claude Code plugin

For a richer experience, install the bundled plugin. It configures the MCP
server for you and adds four skills:

| Skill             | Source     | Description                                        |
| ----------------- | ---------- | -------------------------------------------------- |
| `voip:say`        | MCP server | Call a number, speak a message, and hang up.       |
| `voip:call`       | MCP server | Hold a two-way AI conversation, return transcript. |
| `voip:transcribe` | CLI        | Transcribe incoming call audio to text.            |
| `voip:echo`       | CLI        | Echo the caller's speech back to them.             |

#### Install from the marketplace

The plugin is published in the
[codingjoe/claude-plugins](https://github.com/codingjoe/claude-plugins)
marketplace. Add the marketplace and install the plugin:

```console
/plugin marketplace add codingjoe/claude-plugins
/plugin install voip@codingjoe
```

Then set your `SIP_AOR` — either edit the plugin's `.mcp.json` in the local
plugin cache, or export the `SIP_AOR` environment variable before launching
Claude Code.

### Add the MCP server manually

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

#### Load locally during development

The plugin lives at the repository root. Clone the repo and point Claude Code
at it:

```console
git clone https://github.com/codingjoe/VoIP.git
claude --plugin-dir ./VoIP
```

Then edit the `SIP_AOR` environment variable in the cloned `.mcp.json` to point
to your SIP address-of-record.

## Tools

::: voip.mcp.say

::: voip.mcp.call

[cc-mcp]: https://docs.anthropic.com/en/docs/claude-code/mcp
[mcp]: https://modelcontextprotocol.io/
[plugins]: https://code.claude.com/docs/en/plugins
