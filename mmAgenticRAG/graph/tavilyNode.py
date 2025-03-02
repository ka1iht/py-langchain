from langchain_community.tools import TavilySearchResults

tavily_tool = TavilySearchResults(max_results=2)

tools = [tavily_tool]

tools.invoke("Can you tell me the results of Arsenal's last match?")