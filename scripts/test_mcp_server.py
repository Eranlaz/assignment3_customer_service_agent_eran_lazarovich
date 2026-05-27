import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]


async def main() -> None:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(PROJECT_ROOT / "mcp_server.py")],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()

            print("Available MCP tools:")
            for tool in tools.tools:
                print("-", tool.name)

            result = await session.call_tool(
                "list_categories",
                arguments={},
            )

            print()
            print("list_categories result:")
            for item in result.content:
                print(item)


if __name__ == "__main__":
    asyncio.run(main())
