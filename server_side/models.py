from google.cloud import bigquery
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"Z:/kizX/environments/andrew_env/json/andrew-chronix-d6972bcec0a9.json"

client = bigquery.Client()

def execute_query(query: str):
    """
    Executes a SQL query on BigQuery and returns the results.

    :param query: The SQL query string.
    :return: Query results as a list of dictionaries.
    """
    try:
        query_job = client.query(query)
        results = query_job.result()
        return [dict(row) for row in results]
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
