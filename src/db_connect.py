from sqlalchemy import create_engine
from _env import db_credentials


# PostgreSQL connection parameters
user = db_credentials.user
password = db_credentials.password
host = db_credentials.host
port = db_credentials.port
dbname = db_credentials.dbname


def create_db_engine():
    return create_engine(f'postgresql://{user}:{password}@{host}:{port}/{dbname}')



