from google.cloud import bigquery
import os
from werkzeug.security import generate_password_hash, check_password_hash

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"Z:/kizX/environments/andrew_env/json/andrew-chronix-d6972bcec0a9.json"

client = bigquery.Client()

def create_user(username, name, email_id, password):
    table_id = "andrew-chronix.andrew_data.users"
    hashed_password = generate_password_hash(password) if password else None

    rows_to_insert = [{
        "username": username,
        "name": name,
        "email_id": email_id,
        "password": hashed_password
    }]

    try:
        errors = client.insert_rows_json(table_id, rows_to_insert)
        if errors:
            print(f"BigQuery Errors: {errors}")
        return errors == []
    except Exception as e:
        print(f"Error creating user in BigQuery: {e}")
        return False

def authenticate_user(email_id, password):
    table_id = "andrew-chronix.andrew_data.users"
    query = f"SELECT password FROM `{table_id}` WHERE email_id = @email_id"

    try:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("email_id", "STRING", email_id)]
        )
        results = client.query(query, job_config=job_config).result()

        for row in results:
            if check_password_hash(row["password"], password):
                return True
        return False
    except Exception as e:
        print(f"Error authenticating user: {e}")
        return False