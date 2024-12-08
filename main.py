from langchain_ollama.llms import OllamaLLM # to use Ollama llms in langchain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate # crafts prompts for our llm
from langchain_core.tools import tool # tools for our llm
from langchain.tools.render import render_text_description # to describe tools as a string 
from langchain_core.output_parsers import JsonOutputParser # ensure JSON input for tools
from operator import itemgetter # to retrieve specific items in our chain.
import requests # to get the data for our RAG
import os
import json

OLLAMA_MODEL = 'hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF'

SYSTEM_PROMPT = """You are an assistant that has access to the following set of tools. 
You should only answer with a JSON blob as output and nothing else.
Here are the names and descriptions for each tool:

{rendered_tools}

Given the user input, return the name and input of the tool to use.
Return your response as a JSON blob with 'name' (string) and 'arguments' (array) keys. 
The value associated with the 'arguments' key should be a dictionary of parameters.
"""

USER_PROMPT = """Use the following JSON to answer the user question in the end. 
Never mention the JSON to the user just use it to answer the user question.

{tool_output}

User question: {input}
"""

model = OllamaLLM(model=OLLAMA_MODEL)

LOG = os.getenv("LOG", "INFO")

def debug_log(title: str, log: str|dict):
    if LOG == "DEBUG":
        debug_txt = '\x1b[7;30;44mDEBUG \x1b[0m'
        parsed_log = json.dumps(log, indent=2) if type(log) == dict else log;
        print(f"{debug_txt} # {title}: \n{debug_txt} | {parsed_log.replace("\n", f"\n{debug_txt} | ")}")

def error_dict(error_message: str) -> dict:
    return {
        "error": error_message
    }

@tool
def get_weather(location: str) -> dict:
    "Gets the weather for specific location."
    resp = requests.get(f'https://wttr.in/{location}?format=j2')
    if resp.status_code >= 400 and resp.status_code < 500:
        return error_dict(f"Location '{location}' does not exist.")
    elif resp.status_code > 500:
        return error_dict("Server is experiencing some problems, could not retrieve weather for location.")
    resp = resp.json()
    debug_log("TOOL RESPONSE", resp)
    return resp

@tool
def convert_currency(source_currency: str):
    "Gets currency rate for source_currency."
    resp = requests.get(f"https://open.er-api.com/v6/latest/{source_currency}")
    if resp.status_code >= 400 and resp.status_code < 500:
        return error_dict(f"Currency '{source_currency}' not found.")
    elif resp.status_code > 500:
        return error_dict("Server is experiencing some problems, could get conversion rates for currency.")
    resp = resp.json()
    debug_log("TOOL RESPONSE", resp)
    return resp

@tool
def add(first: int, second: int) -> dict:
    "Add two integers."
    resp = { "result": first + second }
    debug_log("TOOL RESPONSE", resp)
    return resp

tools = [get_weather, add, convert_currency]

def invoke_tool(model_output):
    debug_log("TOOL INVOKED", model_output)
    tool_map = {tool.name: tool for tool in tools}
    chosen_tool = tool_map[model_output["name"]]
    return chosen_tool.run(model_output["arguments"])

if __name__ == "__main__":
    rendered_tools = render_text_description(tools)
    system_prompt = SYSTEM_PROMPT.format(rendered_tools=rendered_tools)

    debug_log("SYSTEM_PROMPT", system_prompt)
    
    prompt = ChatPromptTemplate([
        ("system", system_prompt), 
        ("user", "{input}")
    ])
    
    result_prompt = PromptTemplate.from_template(USER_PROMPT)

    tool_chain = prompt | model | JsonOutputParser() | invoke_tool
    llm_chain = result_prompt | model
    chain = ({
        "tool_output": tool_chain,
        "input": itemgetter("input")
    } | llm_chain)

    query = input("> ")
    while query not in ['exit', 'quit']:
        response = chain.invoke({'input': query})
        print(response)
        query = input("> ")
