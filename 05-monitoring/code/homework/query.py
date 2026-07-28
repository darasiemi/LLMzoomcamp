import sqlite3
import pandas as pd

conn = sqlite3.connect("traces.db")

query_5 = """
SELECT
    name,
    ROUND(SUM(end_time - start_time) / 1000000.0, 3) AS total_duration_ms
FROM spans
WHERE name NOT IN ('rag', 'homework5')
GROUP BY name
ORDER BY total_duration_ms DESC;
"""

query_6 = """
SELECT
    ROW_NUMBER() OVER (ORDER BY start_time) AS rag_call,
    input_tokens
FROM spans
WHERE name = 'llm'
ORDER BY start_time;
"""

df = pd.read_sql(query_6, conn)

print(df)