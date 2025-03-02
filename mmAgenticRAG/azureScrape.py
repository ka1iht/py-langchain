from langchain_community.document_loaders import RecursiveUrlLoader

loader = RecursiveUrlLoader("https://learn.microsoft.com/en-us/azure/architecture/browse/")

docs = loader.load()

print(docs)