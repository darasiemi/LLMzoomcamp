For this project, we will use `uv`, the environment manager because it is fast and very useful.

After install `uv`, initialize it,
```bash
uv init
```

Add the dependencies we'll need. We will be using Google AI studio,
```bash
uv add requests minsearch sqlitesearch jupyter python-dotenv google-genai
```

This installs:

- requests - to fetch the FAQ dataset from the internet
- minsearch - a simple in-memory search engine for indexing and searching text
- google-genai - the Google AI studio API client for calling the LLM
- jupyter - the notebook environment where we'll write and run code
- python-dotenv - to load API keys from a .env file

## Starting Jupyter
Start Jupyter:

```bash
uv run jupyter notebook
```
