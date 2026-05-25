# Assignment 3 - Customer Service Data Analyst Agent

LangGraph-based customer service data analyst agent for the Bitext Customer Service dataset.

This project currently implements Task 1: an initial ReAct-style data analyst agent that answers structured and unstructured questions about the dataset, while politely declining out-of-scope questions.

## Current Features

- Loads and analyzes the Bitext Customer Service dataset.
- Classifies user queries as structured, unstructured, or out-of-scope.
- Uses LangGraph to run a ReAct-style agent loop.
- Uses dataset tools with Pydantic input schemas.
- Prints tool calls and observations in the CLI.
- Includes a max-iteration fallback to avoid infinite loops.
- Uses Nebius Token Factory for LLM calls.

## Project Structure

```text
.
├── main.py
├── requirements.txt
├── data/
│   └── bitext_customer_service.csv
├── scripts/
│   ├── explore_dataset.py
│   ├── test_dataset_functions.py
│   ├── test_graph.py
│   ├── test_llm.py
│   ├── test_router.py
│   └── test_tools.py
└── src/
    ├── config.py
    ├── data_loader.py
    ├── dataset_functions.py
    ├── graph.py
    ├── llm.py
    ├── router.py
    └── tools.py
