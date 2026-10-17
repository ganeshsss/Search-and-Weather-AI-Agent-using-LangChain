
from langchain.tools import tool


import requests 


from langchain_core.prompts import PromptTemplate


from langchain_tavily import TavilySearch


##pip install --force-reinstall --no-deps python-dotenv


import dotenv
print(dotenv.__file__)



#pip install langchain-community langchain langchain-classic tavily-python



#pip install python-dotenv langchain-google-genai certifi


import sys
print(sys.executable)
#pip show langchain-google-genai


import os
import certifi
from dotenv import load_dotenv

#from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_classic import hub


from langchain_classic.agents import create_react_agent, AgentExecutor


#================================
# LOAD ENVIRONMENT VARIABLES
#================================
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY= os.getenv("WEATHERSTACK_API_KEY")


search_tool=TavilySearch(max_results=2)


@tool
def get_weatheer_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """
    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    response = requests.get(url)
    data = response.json()

    if "current" not in data:
        return f"Weather data not available for {city}."

    # Weatherstack returns a list like ["Patchy rain nearby"]. 
    # Grab the first element safely without character-joining bugs:
    weather_desc = data['current']['weather_descriptions']
    if isinstance(weather_desc, list) and len(weather_desc) > 0:
        desc_str = weather_desc[0]
    else:
        desc_str = str(weather_desc)

    return (
        f"city: {city}\n"
        f"temperature: {data['current']['temperature']}°C\n"
        f"weather_descriptions: {desc_str}\n"
        f"Humidity: {data['current']['humidity']}#"
    )
    


result=search_tool.invoke("Give me the latest news of IND VS WI  17/10/2026")
result


#===============================
# LLM
#===============================
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",  # Use an OpenRouter model name (or any other free/paid model)
    temperature=0,
    google_api_key=GEMINI_API_KEY   
)


response=llm.invoke("what year is it?")
response


# Define the standard ReAct template locally
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


#===============================
# CREATE TOOLS
#===============================
tools=[search_tool,get_weatheer_data]


#===============================
# CREATE AGENT
#===============================
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)


#===============================
# EXECUTOR
#===============================
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True)


#===============================
# RUN
#===============================
response=agent_executor.invoke({
    "input":(
        "The capital of France is Paris. What is the capital of Germany?"
        " and what is the weather in Berlin?" 
    )
})

print(response['output'])


