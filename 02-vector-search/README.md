To install the sentence transformer
```bash
uv add sentence-transformers
```
Install psycopg
```bash
uv add psycopg 
```
Start pgvector
```bash
docker run -it \
    --name pgvector \
    -e POSTGRES_USER=user \
    -e POSTGRES_PASSWORD=pswd \
    -e POSTGRES_DB=faq \
    -v pgvector_data:/var/lib/postgresql/data \
    -p 5433:5432 \
    pgvector/pgvector:pg17
```