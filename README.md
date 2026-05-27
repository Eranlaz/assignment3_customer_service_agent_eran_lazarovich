# Assignment 3 - Customer Service Data Analyst Agent

LangGraph-based data analyst agent for the Bitext Customer Service dataset.

The agent answers structured and unstructured questions about the dataset, declines out-of-scope questions, supports persistent conversation memory, maintains a persistent user profile, and exposes selected dataset tools through a FastMCP server.

## Implemented Tasks

- Task 1: Initial LangGraph data analyst agent.
- Task 2a: Persistent conversation memory with SQLite checkpointer.
- Task 2b: Persistent user profile.
- Task 3: FastMCP server exposing dataset tools.

## Setup

Create and activate a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

Create a local .env file:

    cp .env.example .env

Then edit .env:

    NEBIUS_API_KEY=your_nebius_api_key_here
    NEBIUS_MODEL=meta-llama/Llama-3.3-70B-Instruct

Do not commit .env.

## Run the CLI Agent

Basic run:

    python main.py

Run with persistent conversation memory:

    python main.py --session my_session

Run with a persistent user profile:

    python main.py --session my_session --user my_user

Exit with:

    exit

## Example Queries

Structured:

    What categories exist in the dataset?
    How many refund requests did we get?
    Show me 5 examples of the SHIPPING category.
    What is the distribution of intents in the ACCOUNT category?
    Compare the intent distributions of ACCOUNT and REFUND.

Unstructured:

    Summarize the FEEDBACK category.
    How do customer service representatives typically respond to cancellation requests?

Out-of-scope:

    Who is the president of France?
    What's the best CRM software for handling complaints?
    Write me a poem about customer service.

Memory:

    Show me 3 examples from the REFUND category.
    Show me 3 more.

    How many complaints did we get?
    What about refunds?
    What is the total count of the last two?

User profile:

    My name is Eran and I care mostly about refund analysis. I prefer step-by-step explanations.
    What do you remember about me?

## Architecture

The agent uses a LangGraph graph:

    User query
        -> Router node
        -> structured / unstructured / out-of-scope
        -> dataset tools
        -> final answer

Out-of-scope questions are declined before the model answers from general knowledge.

The graph has a max-iteration fallback to avoid infinite loops.

## Model Choice

The project uses Nebius Token Factory through an OpenAI-compatible LangChain client.

Current model:

    meta-llama/Llama-3.3-70B-Instruct

This model was selected because it provides strong instruction following for routing, tool selection, summarization, and final-answer generation.

## Internal Tools

The LangGraph agent uses these tools:

- list_categories
- list_intents
- count_records
- show_examples
- intent_distribution
- search_records

Each tool has a clear description and a Pydantic input schema.

## Persistent Memory

Conversation memory uses LangGraph checkpoints with SQLite.

Run with:

    python main.py --session my_session

The same session ID restores the same conversation after restart.

SQLite checkpoints are stored under:

    storage/

This folder is ignored by Git.

## User Profile

User profiles are stored separately from conversation history.

Run with:

    python main.py --session my_session --user my_user

Profiles are stored under:

    profiles/

This folder is ignored by Git.

## FastMCP Server

The MCP server is implemented in:

    mcp_server.py

It exposes these MCP tools:

- list_categories
- list_intents
- count_dataset_records
- show_dataset_examples
- get_intent_distribution_for_category
- search_dataset_records

Start the MCP server:

    python mcp_server.py

The server uses stdio transport and waits for an MCP client.

## MCP Client Test

A small MCP client is included:

    python scripts/test_mcp_server.py

It starts the MCP server, lists available MCP tools, and calls:

    list_categories

Optional MCP Inspector:

    mcp dev mcp_server.py

## Tests

LLM connection:

    python scripts/test_llm.py

Dataset functions:

    python scripts/test_dataset_functions.py

Tools:

    python scripts/test_tools.py

Router:

    python scripts/test_router.py

Graph:

    python scripts/test_graph.py

MCP server:

    python scripts/test_mcp_server.py
