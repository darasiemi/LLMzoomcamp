```bash
uv add streamlit
```

To run commands from Makefile, e.g. chat, run
```bash
make file
``` 
To add dependency
```bash
uv add "psycopg[binary]"
```
Connect to database
```bash
uv run python db_init.py
```

To check the data
```bash
docker exec -it course-assistant-pg psql -U user -d course_assistant \
    -c "SELECT id, question, response_time, cost FROM conversations;"
```

To display Streamlit dashboard
```bash
uv run streamlit run dashboard.py --server.port 8502
```

To run the built-in judge
```bash
uv run python judge.py
```

Generate synthetic data to evaluate the model
```bash
uv run python generate_data.py
```

To run grafana
```bash
docker run -d \
    --name grafana \
    --network monitoring \
    -p 3001:3000 \
    -v grafana_data:/var/lib/grafana \
    grafana/grafana
```

Query for response time
```bash
SELECT
  timestamp AS time,
  response_time
FROM conversations
WHERE timestamp BETWEEN $__timeFrom() AND $__timeTo()
ORDER BY timestamp
```
Query for token usage
```bash
SELECT
  $__timeGroup(timestamp, $__interval) AS time,
  AVG(total_tokens) AS avg_tokens
FROM conversations
WHERE timestamp BETWEEN $__timeFrom() AND $__timeTo()
GROUP BY 1
ORDER BY 1
```

Cost Panel
```bash
SELECT
  $__timeGroup(timestamp, $__interval) AS time,
  SUM(cost) AS total_cost
FROM conversations
WHERE timestamp BETWEEN $__timeFrom() AND $__timeTo()
  AND cost > 0
GROUP BY 1
ORDER BY 1
```

Model Usage Panel
```bash
SELECT
  model,
  COUNT(*) as count
FROM conversations
WHERE timestamp BETWEEN $__timeFrom() AND $__timeTo()
GROUP BY model
```

Relevance Distribution Panel
```bash
SELECT
  relevance,
  COUNT(*) as count
FROM feedback
WHERE source = 'judge'
  AND timestamp BETWEEN $__timeFrom() AND $__timeTo()
GROUP BY relevance
```

User Feedback Panel
```bash
SELECT
  SUM(CASE WHEN score > 0 THEN 1 ELSE 0 END) as thumbs_up,
  SUM(CASE WHEN score < 0 THEN 1 ELSE 0 END) as thumbs_down
FROM feedback
WHERE source = 'user'
  AND timestamp BETWEEN $__timeFrom() AND $__timeTo()
```

Recent Conversations Panel
```bash
SELECT
  timestamp AS time,
  question,
  answer,
  response_time,
  cost
FROM conversations
WHERE timestamp BETWEEN $__timeFrom() AND $__timeTo()
ORDER BY timestamp DESC
LIMIT 5
```