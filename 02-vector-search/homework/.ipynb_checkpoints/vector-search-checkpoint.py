import numpy as np
from embedder import Embedder
from gitsource import GithubRepositoryDataReader
from gitsource import chunk_documents
from minsearch import VectorSearch
from minsearch import Index

def download_lessons():
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )

    documents = [file.parse() for file in reader.read()]

    return documents

def keyword_search(query,chunks, num_results=5):
    index = Index(
        text_fields=['content'],
        keyword_fields=['filename']
    )
    index.fit(chunks)

    boost_dict = {
        "filename": 1.5,
        "content": 1.0,
    }

    return index.search(
        query,
        num_results=num_results,
        boost_dict=boost_dict,
    )


def vec_search(X, chunks):
    
    vindex = VectorSearch(keyword_fields=["course"])
    vindex.fit(X, chunks)

    return vindex

def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]

if __name__ == "__main__":

    model = Embedder()

    q1 = "How does approximate nearest neighbor search work?"
    v1 = model.encode(q1)

    print(f"first similarity of vect1 {v1[0]}")

    documents = download_lessons()
    target = "02-vector-search/lessons/07-sqlitesearch-vector.md"

    doc = next(d for d in documents if d["filename"] == target)

    v2 = model.encode(doc["content"])

    print(np.linalg.norm(v1))
    print(np.linalg.norm(v2))

    print(f"Dot product {v2.dot(v1)}")


    chunks = chunk_documents(documents, size=2000, step=1000)

    texts = [chunk["content"] for chunk in chunks]

    X = model.encode_batch(texts)

    scores = X.dot(v1)

    best_idx = np.argmax(scores)
    best_chunk = chunks[best_idx]

    print(best_chunk["filename"])
    # print(best_chunk["content"])

    q3 = "What metric do we use to evaluate a search engine?"

    # v3 = model.encode(q3)

    q4 = "How do I store vectors in PostgreSQL?"

    vindex= vec_search(X, chunks)
    # vindex = VectorSearch(keyword_fields=["course"])
    # vindex.fit(X, chunks)

    v4 = model.encode(q4)

    results = vindex.search(v4, num_results=5)

    print("Vector search----------------------------------")

    for i in range(len(results)):
        print(results[i]["filename"])

    print("Keyword search-------------------------------------")

    keyword_results = keyword_search(q4,chunks, num_results=5)

    for i in range(len(results)):
        print(keyword_results[i]["filename"])

    q5 = "How do I give the model access to tools?"

    v5 = model.encode(q5)

    vector_results = vindex.search(v5, num_results=5)

    print("Q6---Vector search----------------------------------")

    for i in range(len(vector_results)):
        print(vector_results[i]["filename"])

    print("Q6-----Keyword search-------------------------------------")

    text_results = keyword_search(q5,chunks, num_results=5)

    for i in range(len(text_results)):
        print(text_results[i]["filename"])

    results = rrf([vector_results,  text_results])
    
    print("Hybrid search------------------------------------")
    for result in results:
        print(result["filename"])

    # print(results[0]["filename"])

    # cosine_similarity = np.dot(v1, v2) / (
    # np.linalg.norm(v1) * np.linalg.norm(v2)
    # )

    # print("Cosine similarity", cosine_similarity)

    