"""Starter code for the monitoring homework.

Sets up the text-search RAG from homework 1 and a shared OpenAI client.
"""

from google import genai

from gitsource import GithubRepositoryDataReader
from minsearch import Index

from rag_helper import GeminiRAGTraced

from dotenv import load_dotenv
load_dotenv()

COMMIT = "8c1834d"

# --- Load the course lessons (same as HW1, HW2, HW4) ---
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id=COMMIT,
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]

print(documents[0])

index = Index(text_fields=["content"], keyword_fields=["filename"])
index.fit(documents)

client = genai.Client()
rag = GeminiRAGTraced(
    index=index,
    llm_client=client,
    model="gemini-2.5-flash",
)





if __name__ == "__main__":
    # print(documents[0])
    query = "How does the agentic loop keep calling the model until it stops?"
    answer = rag.rag(query)
    print(answer)
