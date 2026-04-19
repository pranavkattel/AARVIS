import os
import re
from langchain_core.messages import SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from app.agent.state import AgentState
from app.agent.tools import tools
from datetime import datetime, timedelta
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Bind all tools to Ollama
#model = ChatOllama(model="qwen3:4b", temperature=0.7).bind_tools(tools)

# model = ChatGoogleGenerativeAI(
#     model="gemini-3-flash-preview",
#     temperature=0.7,
#     google_api_key=os.getenv("GEMINI_API_KEY")
# ).bind_tools(tools)

model = ChatGroq(
    model="openai/gpt-oss-20b",  # Changed to a valid Groq model name; change to what you need
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.6,
    max_tokens=1024,
).bind_tools(tools)

SYSTEM_PROMPT_TEMPLATE = """/no_think
You are AARVIS, a smart mirror voice assistant. You speak naturally and concisely.
Today is {today_date}. Tomorrow is {tomorrow_date}.

You have tools for: calendar, email, weather, news.

STRICT RULES YOU MUST FOLLOW:
1. For greetings like "hello", "hi", "hey", "good morning" — just reply naturally. DO NOT call any tools.
2. NEVER call create_calendar_event, delete_calendar_event, update_calendar_event, or draft_and_send_email unless the user EXPLICITLY asks you to create/delete/update/send something AND you have confirmed with them.
3. Call each tool at most ONCE per turn.
4. When calling create_calendar_event, use 24-hour HH:MM format (e.g. "14:00" not "2:00 PM").
5. Never output raw JSON. Always speak in natural sentences.
6. Keep responses under 3 sentences — they are spoken aloud. Exception: for news results, you may provide a short numbered list with details.
7. If the user asks about their schedule/meetings, call get_calendar_today. Do NOT create anything.
8. If unsure what the user wants, ASK — do not guess.
9. Calendar tools return event_id values. Remember them — you need event_id to update or delete events.
10. Do NOT wrap your response in <think> tags or output any internal reasoning.
11. For ALL email sending requests — whether the user wants you to write it for them ('write an email to John saying he is fired') OR provides explicit content — use send_email. If the user only gives a topic, leave subject/body empty and it will auto-compose. If they give you the full content, pass subject and body directly.
12. If the user asks for news, headlines, or current events, call get_news exactly once in that turn.
13. Default to world headlines (use get_news with personalized=false) unless the user explicitly asks for personalized/topic/country news.
13a. For country-specific requests (example: "Nepal news"), call get_news with country set to that country and personalized=true.
14. If the user asks to save or change their news interests/country, call set_news_preferences. If they also ask for personalized headlines now, call get_news with personalized=true.
15. If the user explicitly asked for a specific country (for example Nepal), do NOT suggest world headlines unless the user asks to switch to world news."""


def model_call(state: AgentState) -> AgentState:
    """Central LLM node — follows agent3 pattern exactly."""
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d (%A)")
    tomorrow_str = (now + timedelta(days=1)).strftime("%Y-%m-%d (%A)")

    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        today_date=today_str,
        tomorrow_date=tomorrow_str,
    )

    # Add user context
    user_context = f"\nCurrent user: {state.get('current_user', 'User')}"
    if state.get('user_location'):
        user_context += f"\nUser's location: {state['user_location']}"
    if state.get('user_interests'):
        user_context += f"\nUser's interests: {state['user_interests']}"

    system = SystemMessage(content=prompt + user_context)
    try:
        response = model.invoke([system] + list(state["messages"]))
    except Exception as exc:
        print(f"[AARVIS] Model invocation failed: {exc}")

        # Simple local fallback so UI remains usable if Ollama is unavailable/OOM
        last_text = ""
        if state.get("messages"):
            last = state["messages"][-1]
            last_text = (getattr(last, "content", "") or "").strip().lower()

        if any(greet in last_text for greet in ["hi", "hello", "hey", "good morning", "good evening"]):
            fallback = "Hi! I'm here. My local AI model is restarting, but I can try again in a moment."
        else:
            fallback = "I’m having trouble with the local AI model right now. Please try again in a few seconds."

        return {"messages": [AIMessage(content=fallback)]}

    # Normalize content: Gemini returns a list of parts, Ollama returns a string
    raw_content = response.content or ""
    if isinstance(raw_content, list):
        # Extract text from Gemini-style content parts
        parts = []
        for part in raw_content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
            elif hasattr(part, "text"):
                parts.append(part.text)
        raw_content = "".join(parts)

    # Strip <think>...</think> blocks from qwen3 output (handles unclosed tags too)
    cleaned_content = raw_content
    cleaned_content = re.sub(r"<think>[\s\S]*?</think>", "", cleaned_content)  # closed tags
    cleaned_content = re.sub(r"<think>[\s\S]*$", "", cleaned_content)            # unclosed tag (truncated output)
    cleaned_content = cleaned_content.strip()

    # Deduplicate tool calls by name — qwen3 sometimes emits the same tool twice
    # with slightly different args. Keep only the most complete call per tool name.
    deduped_calls = []
    if hasattr(response, 'tool_calls') and response.tool_calls:
        seen_tools = {}
        for tc in response.tool_calls:
            name = tc['name']
            args = tc.get('args', {})
            if name not in seen_tools or len(args) > len(seen_tools[name].get('args', {})):
                seen_tools[name] = tc
        deduped_calls = list(seen_tools.values())

    response = AIMessage(
        content=cleaned_content,
        tool_calls=deduped_calls,
        response_metadata=response.response_metadata if hasattr(response, 'response_metadata') else {},
    )

    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Routing function — mirrors agent3_with_task.py exactly."""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    return "continue"


# Build graph — identical structure to agent3_with_task.py
graph = StateGraph(AgentState)
graph.add_node("aarvis_agent", model_call)
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("aarvis_agent")
graph.add_conditional_edges(
    "aarvis_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)
graph.add_edge("tools", "aarvis_agent")  # loop back after tool execution

# Compile with recursion limit to prevent infinite tool loops
agent = graph.compile()
agent.recursion_limit = 10
