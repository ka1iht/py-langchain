# ------------------------------------------------------
# Streamlit
# Knowledge Bases for Amazon Bedrock and LangChain 🦜️🔗
# ------------------------------------------------------

import boto3
import logging


from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_aws import ChatBedrockConverse
from langchain_aws import AmazonKnowledgeBasesRetriever
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from typing import Any
from langchain_core.callbacks import AsyncCallbackHandler

from langchain_core.tools import tool

import requests
from tabulate import tabulate

class BedrockAsyncCallbackHandler(AsyncCallbackHandler):
    # Async callback handler that can be used to handle callbacks from langchain.

    async def on_llm_error(self, error: BaseException, **kwargs: Any) -> Any:
        reason = kwargs.get("reason")
        if reason == "GUARDRAIL_INTERVENED":
            print(f"Guardrails: {kwargs}")

# Tool functions
@tool
def getWeather(city: str) -> str:
    """Get the current weather in a city.

    Args: city (str): The city to get the weather for.

    Returns: str: The current weather in the city.
    """

    # debug
    print("The model has called the getWeather function!")

    response = requests.get(f"http://api.weatherapi.com/v1/current.json?key=87283b266dfc4a1e9f3170118251202&q={city}&aqi=no")

    return f"The current weather in {response.json()["location"]["name"]}, {response.json()["location"]["region"]} is {response.json()["current"]["condition"]["text"]}. The current temperature is {response.json()["current"]["temp_f"]} F, and feels like {response.json()["current"]["feelslike_f"]} F."

@tool
def getAccountName() -> str:
    """Get the name of the AWS account running this model.

    Returns: str: The name of the AWS account running this model.
    """

    # debug
    print("The model has called the getAccountName function!")

    return "The current AWS account running this model is: " + boto3.client("sts").get_caller_identity()["Account"] + " and the user authenticated is: " + boto3.client("sts").get_caller_identity()["UserId"]

@tool
def getInstances(region: str) -> str:
    """Get the current AWS EC2 instances in a specified region.

    Args: region (str): The region to get the EC2 instances for.

    Returns: str: The list of the current instances in the region.
    """

    # debug
    print("The model has called the getInstances function!")

    instances = []

    reservationDict = boto3.client("ec2", region_name=region).describe_instances()["Reservations"]

    if len(reservationDict) > 0:
        for instance in reservationDict:
            for i in instance["Instances"]:
                instances.append([i["InstanceId"], i["PrivateIpAddress"], i["State"]["Name"]])
        return "The current instances in the region " + region + " are:\n" + tabulate(instances, headers=["Instance ID", "Private IP", "State"], tablefmt="github")
    
    else:
        return "There are no instances in the region " + region + "."

@tool
def createInstance(name: str, region: str, ami_id: str, instance_type: str) -> str:
    """Create an AWS EC2 instance in a specified region.

    Args: name (str): The name to give to the EC2 instance.
          region (str): The region to create the EC2 instance in.
          ami_id (str): The AMI ID to use for the instance.
          instance_type (str): The instance type to use for the instance.

    Returns: str: The ID of the created instance.
    """

    # debug
    print("The model has called the createInstance function!")

    response = boto3.client("ec2", region_name=region).run_instances(
        ImageId=ami_id,
        InstanceType=instance_type,
        MinCount=1,
        MaxCount=1
    )

    boto3.client("ec2", region_name=region).create_tags(Resources=[response["Instances"][0]["InstanceId"]], Tags=[{'Key':'Name', 'Value': name}])

    return "The EC2 instance has been created in the " + region + " with the ID: " + response["Instances"][0]["InstanceId"]

# Tools
tools = [getWeather, getAccountName, getInstances, createInstance]

# ------------------------------------------------------
# Log level

logging.getLogger().setLevel(logging.ERROR) # reduce log level

# ------------------------------------------------------
# Amazon Bedrock - settings

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

# ------------------------------------------------------
# LangChain - RAG chain with chat history

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."
         "Answer the question based only on the following context:\n {context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# Amazon Bedrock - KnowledgeBase Retriever
retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="EACIXSYKFN", # 👈 Set your Knowledge base ID
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
)

model = ChatBedrockConverse(
    client=bedrock_runtime,
    model_id=model_id,
    max_tokens=2048,
    temperature=0.2,
    top_p=0.9,
    guardrails={
        "guardrailIdentifier": "rvmgaef4v5qp",
        "guardrailVersion": "DRAFT",
        "trace": "enabled"
    },
    stop_sequences=["\n\nHuman"],
    callbacks=[BedrockAsyncCallbackHandler()]
)

model_with_tools = model.bind_tools(tools)

chain = (
    RunnableParallel({
        "tools": itemgetter("question") | model_with_tools,
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "history": itemgetter("history"),
    })
    .assign(response = prompt | model_with_tools | StrOutputParser())
    .pick(["response", "context", "tools"])
)

# response = model_with_tools.invoke("Get me a list of current instances in the AWS Northern Virginia region.")
# if response.tool_calls:
#     for tool_call in response.tool_calls:
#         selected_tool = {"getweather": getWeather, "getaccountname": getAccountName, "getinstances": getInstances}[tool_call["name"].lower()]
#         toolResponse = selected_tool.invoke(tool_call)
#         print(toolResponse.content)

# Streamlit Chat Message History
history = StreamlitChatMessageHistory(key="chat_messages")

# Chain with History
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="question",
    history_messages_key="history",
    output_messages_key="response",
)
