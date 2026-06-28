from gitsource import GithubRepositoryDataReader
from minsearch import Index
from .rag_helper import HWRAG
from .llm_client import client
from gitsource import chunk_documents


USER_PROMPT_TEMPALATE = '''
Question:
{question}

Context:
{context}
'''

INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''

def build_context(search_results):
    lines = []

    for doc in search_results:
        lines.append(doc['section'])
        lines.append('Q: ' + doc['question'])
        lines.append('A: ' + doc['answer'])
        lines.append('')

    return '\n'.join(lines).strip()

if __name__ == "__main__":

    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )

    files = reader.read()

    documents = []

    for file in files:
        doc = file.parse()
        documents.append(doc)

    print(f"The number of documents are {len(documents)}")

    # print(documents)


    index = Index(
        text_fields=['content'],
        keyword_fields=['filename']
    )
    index.fit(documents)

    query = "How does the agentic loop keep calling the model until it stops?"

    # print(index.search(
    #     query,
    #     num_results=1
    # ))

    assistant = HWRAG(index, client)

    question = "How does the agentic loop keep calling the model until it stops?"

    response = assistant.rag(question)

    # print(response.usage_metadata.prompt_token_count)
    # print(response.usage_metadata.candidates_token_count)
    # print(response.usage_metadata.total_token_count)


    chunks = chunk_documents(documents, size=2000, step=1000)

    print(f"The number of chunks are: {len(chunks)}")

    chunk_index = Index(
    text_fields=["content"],
    keyword_fields=["filename"]
    )

    chunk_index.fit(chunks)

    print(f"Number of chunks: {len(chunks)}")

    assistant = HWRAG(
    index=chunk_index,
    llm_client=client
    )

    question = "How does the agentic loop keep calling the model until it stops?"

    response = assistant.rag(question)

    print(response.text)

    print("Prompt tokens:",
      response.usage_metadata.prompt_token_count)

    print("Output tokens:",
        response.usage_metadata.candidates_token_count)

    print("Total tokens:",
        response.usage_metadata.total_token_count)
    
    q3_tokens = 8841

    q5_tokens = response.usage_metadata.prompt_token_count

    print(f"Q3 tokens: {q3_tokens}")
    print(f"Q5 tokens: {q5_tokens}")
    print(f"Reduction factor: {q3_tokens / q5_tokens:.2f}x")

    # prompt = USER_PROMPT_TEMPALATE.format(
    #     question=query,
    #     context=context
    # )

