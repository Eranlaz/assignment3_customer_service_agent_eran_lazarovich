import ast
import json
import uuid
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.dataset_functions import get_categories
from src.llm import get_chat_model
from src.router import QueryType, route_query
from src.tools import DATASET_TOOLS


MAX_AGENT_ITERATIONS = 10


class AgentState(TypedDict):
    """
    State for the customer service data analyst agent.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    route: str
    route_reason: str
    iterations: int


SYSTEM_PROMPT = """
You are a data analyst agent for the Bitext Customer Service dataset.

Your job:
- Answer only questions about the dataset.
- Use tools to inspect the dataset before answering.
- Do not answer from general knowledge.
- If the question asks for counts, categories, examples, intents, or distributions, use the structured tools.
- If the question asks for summaries or qualitative patterns, first retrieve relevant dataset records using tools, then summarize only from those records.
- If a user gives a natural-language phrase, such as "people wanting their money back", use search_records or list_intents to map it to dataset categories/intents.
- For questions involving multiple categories or intents, call only one tool at a time, then continue with the next tool if needed.
- For comparison questions, retrieve data for each side separately before writing the comparison.
- For total-count questions across multiple groups, count each group separately and then add the counts.
- After a tool returns data that answers the question, write the final answer. Do not call the same tool again with the same arguments.
- When asked to show examples, call show_examples once and then display the examples in your answer.
- Never print JSON that describes tool calls. If you need a tool, call the tool through the tool-calling interface.
- Keep answers clear and concise.
- Mention when an answer is based on a sample rather than the full dataset.

Dataset columns:
- instruction: the customer query
- category: the high-level category
- intent: the specific customer intent
- response: the customer service representative response

Useful category names include:
ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION.
"""


FINAL_ANSWER_PROMPT = """
You are writing the final answer for a customer service dataset analysis.

Use only the tool observations provided below.
Do not call tools.
Do not mention internal implementation details.
If examples are provided, show them clearly and compactly.
If counts or distributions are provided, include the numbers.
For comparisons, compare the observed counts and percentages clearly.
"""


def _last_human_message(messages: list[BaseMessage]) -> HumanMessage:
    """
    Return the latest human message from the graph state.
    """
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message

    raise ValueError("No human message found in graph state.")


def _normalize_tool_arg(value: Any) -> Any:
    """
    Normalize tool-call arguments so duplicate detection works even when
    the model changes 5 to "5" or None to "null".
    """
    if isinstance(value, dict):
        return {key: _normalize_tool_arg(inner_value) for key, inner_value in value.items()}

    if isinstance(value, list):
        return [_normalize_tool_arg(item) for item in value]

    if isinstance(value, str):
        text = value.strip()
        lowered = text.lower()

        if lowered in {"", "none", "null", "nil", "n/a", "na"}:
            return None

        if text.isdigit():
            return int(text)

        return text

    return value


def _tool_signature(tool_call: dict[str, Any]) -> str:
    """
    Create a stable signature for detecting duplicate tool calls.
    """
    name = tool_call.get("name", "")
    args = _normalize_tool_arg(tool_call.get("args", {}))

    return f"{name}:{json.dumps(args, sort_keys=True, default=str)}"


def _is_duplicate_tool_call(
    response: AIMessage,
    previous_messages: list[BaseMessage],
) -> bool:
    """
    Return True if the model is trying to repeat a tool call that already happened.
    """
    if not response.tool_calls:
        return False

    previous_signatures = set()

    for message in previous_messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                previous_signatures.add(_tool_signature(tool_call))

    for tool_call in response.tool_calls:
        if _tool_signature(tool_call) in previous_signatures:
            return True

    return False


def _parse_json_like_text(text: str) -> Any:
    """
    Parse text that may be JSON or Python-literal-like JSON.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None


def _coerce_textual_tool_calls(message: AIMessage) -> AIMessage:
    """
    Some models print tool-call JSON as text instead of emitting real tool calls.
    This converts simple textual tool-call descriptions into real AIMessage tool calls.
    """
    if message.tool_calls:
        return message

    if not isinstance(message.content, str):
        return message

    text = message.content.strip()

    if "function" not in text and "parameters" not in text and "name" not in text:
        return message

    parsed = _parse_json_like_text(text)
    if parsed is None:
        return message

    if isinstance(parsed, dict):
        candidates = [parsed]
    elif isinstance(parsed, list):
        candidates = parsed
    else:
        return message

    valid_tool_names = {tool.name for tool in DATASET_TOOLS}
    tool_calls = []

    for index, candidate in enumerate(candidates):
        if isinstance(candidate, str):
            candidate = _parse_json_like_text(candidate)

        if not isinstance(candidate, dict):
            continue

        name = candidate.get("name")
        args = (
            candidate.get("args")
            or candidate.get("arguments")
            or candidate.get("parameters")
            or {}
        )

        if isinstance(args, str):
            args = _parse_json_like_text(args) or {}

        if name not in valid_tool_names:
            continue

        if not isinstance(args, dict):
            args = {}

        tool_calls.append(
            {
                "name": name,
                "args": args,
                "id": f"coerced_tool_call_{index}_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }
        )

    if not tool_calls:
        return message

    # Keep only one tool call because the Nebius model supports a single tool call per turn.
    return AIMessage(content="", tool_calls=[tool_calls[0]])


def _mentioned_categories(query: str) -> list[str]:
    """
    Return dataset categories explicitly mentioned in the user query.
    """
    normalized_query = query.lower()
    categories = []

    for category in get_categories():
        if category.lower() in normalized_query:
            categories.append(category)

    return categories


def _completed_categories_for_tool(state: AgentState, tool_name: str) -> set[str]:
    """
    Return categories for which a tool has already been called.
    """
    completed = set()

    for message in state["messages"]:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.get("name") != tool_name:
                    continue

                args = tool_call.get("args") or {}
                category = args.get("category")

                if isinstance(category, str):
                    completed.add(category.upper())

    return completed


def _new_tool_call(name: str, args: dict[str, Any]) -> AIMessage:
    """
    Create an AIMessage containing a single tool call.
    """
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": f"manual_tool_call_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }
        ],
    )



def _manual_conversation_memory_response_if_needed(state: AgentState) -> AIMessage | None:
    """
    Deterministically answer simple conversation-memory questions.

    This is used for Task 2 episodic memory checks, such as:
    "What did I ask before?"
    """
    query = _last_human_message(state["messages"]).content
    normalized_query = query.lower()

    memory_phrases = [
        "what did i ask before",
        "what did i ask",
        "what was my previous question",
        "previous question",
        "what did we discuss",
        "what have we discussed",
        "earlier",
        "last question",
    ]

    if not any(phrase in normalized_query for phrase in memory_phrases):
        return None

    human_messages = [
        message.content
        for message in state["messages"]
        if isinstance(message, HumanMessage)
    ]

    # Exclude the current memory question itself.
    previous_questions = human_messages[:-1]

    if not previous_questions:
        return AIMessage(
            content="I do not have earlier user questions stored in this session yet."
        )

    recent_questions = previous_questions[-5:]

    lines = ["Earlier in this session, you asked:"]
    for index, question in enumerate(recent_questions, start=1):
        lines.append(f"{index}. {question}")

    return AIMessage(content="\n".join(lines))

def _manual_multistep_response_if_needed(state: AgentState) -> AIMessage | None:
    """
    Deterministically handle common multi-step analytical queries one tool call at a time.

    This avoids Nebius errors caused by models trying to emit multiple tool calls
    in a single response, and improves unstructured summaries by retrieving examples first.
    """
    query = _last_human_message(state["messages"]).content
    normalized_query = query.lower()
    categories = _mentioned_categories(query)

    def has_completed_tool_call(tool_name: str, expected_args: dict[str, Any]) -> bool:
        """
        Return True if a matching tool call already exists in the message history.
        """
        normalized_expected = _normalize_tool_arg(expected_args)

        for message in state["messages"]:
            if not isinstance(message, AIMessage) or not message.tool_calls:
                continue

            for tool_call in message.tool_calls:
                if tool_call.get("name") != tool_name:
                    continue

                actual_args = _normalize_tool_arg(tool_call.get("args") or {})

                matched = True
                for key, expected_value in normalized_expected.items():
                    actual_value = actual_args.get(key)

                    if isinstance(expected_value, str) and isinstance(actual_value, str):
                        if expected_value.upper() != actual_value.upper():
                            matched = False
                            break
                    elif actual_value != expected_value:
                        matched = False
                        break

                if matched:
                    return True

        return False

    asks_for_qualitative_summary = any(
        phrase in normalized_query
        for phrase in [
            "summarize",
            "summary",
            "typically",
            "common themes",
            "patterns",
            "how do",
            "respond",
            "responses",
        ]
    )

    if asks_for_qualitative_summary:
        if categories:
            category = categories[0]

            if not has_completed_tool_call("show_examples", {"category": category}):
                return _new_tool_call(
                    name="show_examples",
                    args={"category": category, "n": 10},
                )

            return _generate_final_answer_from_observations(state)

        intent_hints = [
            ("cancellation", "cancel_order"),
            ("cancel", "cancel_order"),
            ("complaint", "complaint"),
            ("complaints", "complaint"),
            ("refund", "get_refund"),
            ("money back", "get_refund"),
            ("shipping", "change_shipping_address"),
        ]

        for hint, intent in intent_hints:
            if hint in normalized_query:
                if not has_completed_tool_call("show_examples", {"intent": intent}):
                    return _new_tool_call(
                        name="show_examples",
                        args={"intent": intent, "n": 10},
                    )

                return _generate_final_answer_from_observations(state)

    if len(categories) < 2:
        return None

    asks_for_intent_distribution_comparison = (
        ("compare" in normalized_query or "comparison" in normalized_query)
        and ("distribution" in normalized_query or "breakdown" in normalized_query)
        and ("intent" in normalized_query or "intents" in normalized_query)
    )

    if asks_for_intent_distribution_comparison:
        completed = _completed_categories_for_tool(state, "intent_distribution")

        for category in categories:
            if category.upper() not in completed:
                return _new_tool_call(
                    name="intent_distribution",
                    args={"category": category},
                )

        return _generate_final_answer_from_observations(state)

    asks_for_combined_count = (
        ("total" in normalized_query or "combined" in normalized_query)
        and ("count" in normalized_query or "number" in normalized_query or "records" in normalized_query)
    )

    if asks_for_combined_count:
        completed = _completed_categories_for_tool(state, "count_records")

        for category in categories:
            if category.upper() not in completed:
                return _new_tool_call(
                    name="count_records",
                    args={"category": category},
                )

        return _generate_final_answer_from_observations(state)

    return None


def _generate_final_answer_from_observations(state: AgentState) -> AIMessage:
    """
    Generate a final answer from existing tool observations without allowing more tool calls.

    This function handles common structured outputs deterministically so the model
    does not make arithmetic mistakes when combining counts or distributions.
    """
    user_message = _last_human_message(state["messages"]).content
    normalized_query = user_message.lower()

    observations: list[dict[str, Any]] = []
    raw_observations: list[str] = []

    for message in state["messages"]:
        if isinstance(message, ToolMessage):
            raw_content = str(message.content)
            raw_observations.append(raw_content[:4000])

            parsed = _parse_json_like_text(raw_content)
            if isinstance(parsed, dict):
                observations.append(parsed)
            else:
                observations.append({"raw": raw_content})

    if not observations:
        return AIMessage(
            content=(
                "I could not find usable dataset observations to answer the question. "
                "Please ask a more specific question about categories, intents, counts, examples, or summaries."
            )
        )

    summary_keywords = [
        "summarize",
        "summary",
        "typically",
        "common themes",
        "patterns",
        "how do",
        "respond",
        "responses",
    ]

    has_example_observations = any(
        (isinstance(observation.get("examples"), list) and observation.get("examples"))
        or (isinstance(observation.get("results"), list) and observation.get("results"))
        for observation in observations
    )

    if has_example_observations and any(keyword in normalized_query for keyword in summary_keywords):
        observations_text = "\n\n".join(
            f"Observation {index}:\n{observation}"
            for index, observation in enumerate(raw_observations, start=1)
        )

        model = get_chat_model()

        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        FINAL_ANSWER_PROMPT
                        + "\nWrite a qualitative summary based only on the provided examples. "
                        + "Mention common customer needs and how agents typically respond."
                    )
                ),
                HumanMessage(
                    content=(
                        f"User question:\n{user_message}\n\n"
                        f"Tool observations:\n{observations_text}\n\n"
                        "Write the summary."
                    )
                ),
            ]
        )

        return AIMessage(content=str(response.content))

    def preview(text: Any, max_chars: int = 220) -> str:
        text = str(text).replace("\n", " ").strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."

    def label_for_count(observation: dict[str, Any]) -> str:
        category = observation.get("category")
        intent = observation.get("intent")

        if category and intent:
            return f"{category} / {intent}"
        if category:
            return str(category)
        if intent:
            return str(intent)
        return "matching records"

    # Categories.
    for observation in observations:
        categories = observation.get("categories")
        if isinstance(categories, list):
            return AIMessage(
                content="The dataset categories are: " + ", ".join(str(category) for category in categories) + "."
            )

    # Examples.
    for observation in observations:
        examples = observation.get("examples")
        if isinstance(examples, list) and examples:
            category = observation.get("category")
            intent = observation.get("intent")
            title_parts = []

            if category:
                title_parts.append(f"category {category}")
            if intent:
                title_parts.append(f"intent {intent}")

            title = " and ".join(title_parts) if title_parts else "the requested filter"

            lines = [f"Here are {len(examples)} examples from {title}:"]

            for index, example in enumerate(examples, start=1):
                instruction = example.get("instruction", "")
                example_intent = example.get("intent", "")
                response = example.get("response", "")

                lines.append(
                    f"{index}. Customer: \"{instruction}\" "
                    f"(intent: {example_intent}). "
                    f"Agent response: {preview(response)}"
                )

            return AIMessage(content="\n".join(lines))

    # Count observations.
    count_observations = [
        observation
        for observation in observations
        if isinstance(observation.get("count"), int)
    ]

    if count_observations:
        if (
            len(count_observations) >= 2
            and (
                "combined" in normalized_query
                or "total" in normalized_query
                or "together" in normalized_query
            )
        ):
            parts = []
            counts = []

            for observation in count_observations:
                label = label_for_count(observation)
                count = int(observation["count"])
                parts.append(f"{label}: {count}")
                counts.append(count)

            total = sum(counts)
            formula = " + ".join(str(count) for count in counts)

            return AIMessage(
                content=(
                    "Counts found:\n"
                    + "\n".join(f"- {part}" for part in parts)
                    + f"\n\nCombined total: {formula} = {total}."
                )
            )

        if len(count_observations) == 1:
            observation = count_observations[0]
            label = label_for_count(observation)
            count = int(observation["count"])

            return AIMessage(content=f"{label} has {count} records in the dataset.")

    # Intent distributions.
    distribution_observations = [
        observation
        for observation in observations
        if isinstance(observation.get("distribution"), list)
    ]

    if distribution_observations:
        lines = []

        if "compare" in normalized_query or "comparison" in normalized_query:
            lines.append("Intent distribution comparison:")
        else:
            lines.append("Intent distribution:")

        summary_lines = []

        for observation in distribution_observations:
            category = observation.get("category", "Unknown")
            distribution = observation.get("distribution", [])

            total = sum(int(item.get("count", 0)) for item in distribution)
            lines.append(f"\n{category}: {total} records across {len(distribution)} intents.")

            if distribution:
                top_item = max(distribution, key=lambda item: int(item.get("count", 0)))
                summary_lines.append(
                    f"{category} has {len(distribution)} intents; the largest is "
                    f"{top_item.get('intent')} with {top_item.get('count')} records "
                    f"({top_item.get('percentage')}%)."
                )

            for item in distribution:
                lines.append(
                    f"- {item.get('intent')}: {item.get('count')} "
                    f"({item.get('percentage')}%)"
                )

        if len(distribution_observations) >= 2:
            lines.append("\nSummary:")
            lines.extend(f"- {line}" for line in summary_lines)

        return AIMessage(content="\n".join(lines))

    # Fallback for unstructured summaries or unusual tool outputs.
    observations_text = "\n\n".join(
        f"Observation {index}:\n{observation}"
        for index, observation in enumerate(raw_observations, start=1)
    )

    model = get_chat_model()

    response = model.invoke(
        [
            SystemMessage(content=FINAL_ANSWER_PROMPT),
            HumanMessage(
                content=(
                    f"User question:\n{user_message}\n\n"
                    f"Tool observations:\n{observations_text}\n\n"
                    "Write the final answer."
                )
            ),
        ]
    )

    return AIMessage(content=str(response.content))


def router_node(state: AgentState) -> dict[str, str]:
    """
    Classify the user query before any tool selection happens.
    """
    user_message = _last_human_message(state["messages"])
    decision = route_query(user_message.content)

    return {
        "route": decision.query_type.value,
        "route_reason": decision.reason,
    }


def decline_node(state: AgentState) -> dict[str, list[AIMessage]]:
    """
    Politely decline out-of-scope questions.
    """
    message = AIMessage(
        content=(
            "I can only answer questions about the Bitext customer service dataset. "
            "You can ask me about categories, intents, counts, examples, distributions, "
            "or summaries of customer-service patterns in the dataset."
        )
    )

    return {"messages": [message]}


def fallback_node(state: AgentState) -> dict[str, list[AIMessage]]:
    """
    Return a graceful fallback when the agent reaches the iteration limit.
    """
    message = AIMessage(
        content=(
            "I could not complete the analysis within the allowed number of steps. "
            "Please ask a more specific question about dataset categories, intents, "
            "counts, examples, distributions, or summaries."
        )
    )

    return {"messages": [message]}


def agent_node(state: AgentState) -> dict[str, object]:
    """
    Call the LLM with tools bound to it.
    """
    iterations = state.get("iterations", 0) + 1

    memory_response = _manual_conversation_memory_response_if_needed(state)
    if memory_response is not None:
        return {
            "messages": [memory_response],
            "iterations": iterations,
        }

    manual_response = _manual_multistep_response_if_needed(state)
    if manual_response is not None:
        return {
            "messages": [manual_response],
            "iterations": iterations,
        }

    base_model = get_chat_model()

    try:
        model = base_model.bind_tools(DATASET_TOOLS, parallel_tool_calls=False)
    except TypeError:
        model = base_model.bind_tools(DATASET_TOOLS)

    try:
        response = model.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                *state["messages"],
            ]
        )
    except Exception as error:
        if "single tool-calls" in str(error).lower():
            manual_response = _manual_multistep_response_if_needed(state)
            if manual_response is not None:
                return {
                    "messages": [manual_response],
                    "iterations": iterations,
                }

        raise

    response = _coerce_textual_tool_calls(response)

    if isinstance(response, AIMessage) and _is_duplicate_tool_call(response, state["messages"]):
        final_answer = _generate_final_answer_from_observations(state)
        return {
            "messages": [final_answer],
            "iterations": iterations,
        }

    return {
        "messages": [response],
        "iterations": iterations,
    }


def route_after_router(state: AgentState) -> Literal["decline", "agent"]:
    """
    Decide where to go after routing.
    """
    if state["route"] == QueryType.OUT_OF_SCOPE.value:
        return "decline"

    return "agent"


def route_after_agent(state: AgentState) -> Literal["tools", "fallback", "end"]:
    """
    Decide whether the agent should call tools, stop, or fallback.
    """
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        if state.get("iterations", 0) >= MAX_AGENT_ITERATIONS:
            return "fallback"

        return "tools"

    return "end"


def build_graph(checkpointer=None):
    """
    Build and compile the LangGraph ReAct-style customer service data analyst agent.
    """
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("decline", decline_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(DATASET_TOOLS))
    graph.add_node("fallback", fallback_node)

    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "decline": "decline",
            "agent": "agent",
        },
    )

    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "fallback": "fallback",
            "end": END,
        },
    )

    graph.add_edge("tools", "agent")
    graph.add_edge("decline", END)
    graph.add_edge("fallback", END)

    return graph.compile(checkpointer=checkpointer)
