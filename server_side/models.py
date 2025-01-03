import pydoop.hdfs as hdfs

HDFS_HOST = "hdfs://localhost:9000"
HDFS_USER = "your_hadoop_user"
HDFS_PATH = "/user/signup_data.txt"

def save_user_data(name, email):
    """
    Save user data to HDFS.
    """
    user_data = f"{name},{email}\n"
    with hdfs.open(HDFS_PATH, "at", user=HDFS_USER) as f:
        f.write(user_data)
