import os
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper # New wrapper for v0.2
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import JSONLoader

from dotenv import load_dotenv

load_dotenv()


generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())


loader = JSONLoader(file_path='data/stock_news.json', jq_schema='.[]', text_content=False)
documents = loader.load()


generator = TestsetGenerator.from_langchain(
    generator_llm,
    generator_llm, # critic
    generator_embeddings
)

testset = generator.generate_with_langchain_docs(documents, testset_size=10)
testset.to_pandas().to_csv("data/gold_dataset.csv", index=False)
