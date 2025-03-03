# imports
# nodes
import tavilyNode

#aws
import boto3

# python
import datetime
from typing import Annotated
from typing_extensions import TypedDict
import json

# langchain
from langchain_aws import ChatBedrockConverse
from langchain.prompts import ChatPromptTemplate
from langchain_core import ToolMessage

# langgraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# IPython
from IPython.display import Image, display

class State(TypedDict):
    messages: Annotated[list, add_messages]

def qa(state: State):
    return {
        "messages": [claude_with_tools.invoke(state["messages"])]
    }

def graph_stream(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1]["content"])

graph_builder = StateGraph(State)

bedrock_client = boto3.client(
    service_name="bedrock-agent-runtime",
    region_name="us-east-1"
)

claude = ChatBedrockConverse(
    client=bedrock_client,
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    max_tokens=2048,
    temperature=0.2,
    top_p=0.9,
    
    guardrails={
        "guardrailIdentifier": "xxxxxxxxxxxx",
        "guardrailVersion": "DRAFT",
        "trace": "enabled"
    }
)

tools = [tavilyNode.tavily_tool]

claude_with_tools = claude.bind_tools(tools)    

print(tools.invoke("What was the result of the last Arsenal match?"))

date = datetime.datetime.today().strftime("%Y-%m-%d")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Today's date is {date}."
         "You are a supervisor node with access to other agents. You will pick the best agent based on the user's question."),
         ("human", "{question}")
    ]
)

graph_builder.add_node("qa", qa)

graph_builder.add_edge(START, "qa")
graph_builder.add_edge("qa", END)

graph = graph_builder.compile()

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception as e:
    print(e)

while True:
    try:
        user_input = input("User: ")
        if user_input == "exit":
            break
        graph_stream(user_input)
    
    except:
        user_input = input("User: What is your name?")
        print("User: ", user_input)
        graph_stream(user_input)
        break

