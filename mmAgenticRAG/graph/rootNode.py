import boto3
import datetime

from typing import Annotated
from typing_extensions import TypedDict

from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
from langchain.chains import RetrievalQA
from langchain_aws import ChatBedrockConverse
from langchain.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from IPython.display import Image, display

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

bedrock_client = boto3.client(
    service_name="bedrock-agent-runtime",
    region_name="us-east-1"
)

retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="",
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 2}}
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

date = datetime.datetime.today().strftime("%Y-%m-%d")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Today's date is {date}."
         "You are a supervisor node with access to other agents. You will pick the best agent based on the user's question."),
         ("human", "{question}")
    ]
)

chain = RetrievalQA.from_chain_type(
    llm = claude,
    retriever = retriever,
    return_source_documents = True
)

def qa(state: State):
    return {
        "messages": [chain.invoke(state["messages"])]
    }

graph_builder.add_node("qa", qa)

graph_builder.add_edge(START, "qa")
graph_builder.add_edge("qa", END)

graph = graph_builder.compile()

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception as e:
    print(e)