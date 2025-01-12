from google.cloud import bigquery
import os
from werkzeug.security import generate_password_hash, check_password_hash

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"Z:/kizX/environments/andrew_env/json/andrew-chronix-d6972bcec0a9.json"

client = bigquery.Client()

def create_user(username, name, email_id, password):
    """
    Inserts a new user into the BigQuery table.
    Passwords are hashed for security.
    """
    table_id = "your_project_id.your_dataset_id.users"
    hashed_password = generate_password_hash(password)

    rows_to_insert = [
        {
            "username": username,
            "name": name,
            "email_id": email_id,
            "password": hashed_password,
        }
    ]

    try:
        errors = client.insert_rows_json(table_id, rows_to_insert)
        if errors == []:
            return True
        else:
            print(f"Insert errors: {errors}")
            return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def authenticate_user(email_id, password):
    """
    Authenticates a user by email and password.
    """
    table_id = "your_project_id.your_dataset_id.users"
    query = f"""
    SELECT password FROM `{table_id}`
    WHERE email_id = @email_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("email_id", "STRING", email_id),
        ]
    )

    try:
        query_job = client.query(query, job_config=job_config)
        results = query_job.result()
        for row in results:
            if check_password_hash(row["password"], password):
                return True
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False