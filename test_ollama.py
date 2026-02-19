from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="phi3")

response = llm.invoke("Explain sales forecasting in simple words")

print(response)
