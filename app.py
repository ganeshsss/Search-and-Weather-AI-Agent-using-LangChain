import os
import certifi
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_classic.agents import create_react_agent, AgentExecutor

# Set page layout and title
st.set_page_config(page_title="LangChain AI Assistant", page_icon="🤖", layout="wide")

# Setup Certifi SSL & Load Environment Variables
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

# Automatically load hidden keys & settings from environment / .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

# Explicitly map Google API key for AQ-prefixed keys or standard keys
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Set default hidden model version (you can change this string here anytime)
HIDDEN_MODEL_NAME = "gemini-3.8-flash"


# =====================================================================
# AGENT SETUP & TOOL DEFINITIONS
# =====================================================================

@tool
def get_weatheer_data(city: str) -> str:
    """Fetch current weather information for a city."""
    if not WEATHERSTACK_API_KEY:
        return "Weatherstack API Key is missing from the .env configuration."

    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    try:
        response = requests.get(url)
        data = response.json()

        if "current" not in data:
            return f"Weather data not available for {city}."

        weather_desc = data['current']['weather_descriptions']
        if isinstance(weather_desc, list) and len(weather_desc) > 0:
            desc_str = weather_desc[0]
        else:
            desc_str = str(weather_desc)

        return (
            f"city: {city}\n"
            f"temperature: {data['current']['temperature']}°C\n"
            f"weather_descriptions: {desc_str}\n"
            f"Humidity: {data['current']['humidity']}%"
        )
    except Exception as e:
        return f"Error fetching weather data: {e}"


# Build ReAct Prompt Template
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)


def get_agent_executor():
    """Helper to initialize the LLM and Agent Executor."""
    if not GEMINI_API_KEY:
        st.error("Gemini API Key is missing. Please check your `.env` configuration file.")
        st.stop()

    tools = [get_weatheer_data]
    if TAVILY_API_KEY:
        search_tool = TavilySearchResults(max_results=2)
        tools.append(search_tool)

    llm = ChatGoogleGenerativeAI(
        model=HIDDEN_MODEL_NAME,
        temperature=0,
        google_api_key=GEMINI_API_KEY
    )

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True
    )


# =====================================================================
# MAIN CLEAN CHAT INTERFACE (No configuration clutter)
# =====================================================================

st.title("🤖 AI Assistant Workspace")
st.caption("Powered by Gemini, Weatherstack, and Tavily Search")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! Ask me about current weather, news, or general knowledge."}
    ]

# Display existing chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User Prompt Input
if user_input := st.chat_input("Ask something... (e.g., What is the weather in Berlin?)"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Process response with Agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking & processing tools..."):
            try:
                executor = get_agent_executor()
                response = executor.invoke({"input": user_input})
                output_text = response.get("output", "No response generated.")

                # Display intermediate reasoning steps securely behind an expander
                intermediate_steps = response.get("intermediate_steps", [])
                if intermediate_steps:
                    with st.expander("🔍 View Agent Thought Process & Tool Execution", expanded=False):
                        for action, observation in intermediate_steps:
                            st.markdown(f"**Action:** `{action.tool}`")
                            st.markdown(f"**Input:** `{action.tool_input}`")
                            st.markdown(f"**Observation:**\n```\n{observation}\n```")
                            st.divider()

                st.write(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    st.error("❌ Quota limit exceeded (429 Rate Limit). Please try again later or update your `.env` key.")
                else:
                    st.error(f"Execution Error: {err_str}")