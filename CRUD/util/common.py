from urllib.parse import quote_plus
import dotenv
import os
import json


class ENVIRONMENT:
    def __init__(self):
        project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
        dotenv_path = os.path.join(project_dir, '.env')
        dotenv.load_dotenv(dotenv_path)
        self.domain = os.getenv("DOMAIN")
        self.port = os.getenv("PORT")
        self.prefix = os.getenv("PREFIX")
        self.db_user = os.getenv("MYSQL_USER")
        self.db_password  =os.getenv("MYSQL_PASSWORD")
        self.db_host = os.getenv("MYSQL_HOST")
        self.db_port = os.getenv("MYSQL_PORT")
        self.db_name = os.getenv("MYSQL_DB")

    def get_instance(self):
        if not hasattr(self, "_instance"):
            self._instance = ENVIRONMENT()
        return self._instance

    def getDomain(self):
        return self.domain

    def getPort(self):
        return self.port

    def getPrefix(self):
        return self.prefix

    def get_db_uri(self):
        encoded_password = quote_plus(self.db_password)
        return f"mysql://{self.db_user}:{encoded_password}@{self.db_host}:{self.db_port}/{self.db_name}"

env  = ENVIRONMENT().get_instance()
domain = env.getDomain()
port = env.getPort()
prefix = env.getPrefix()
db_uri = env.get_db_uri()


def build_swagger_config_json():
    config_file_path = 'static/config/config.json'

    with open(config_file_path, 'r') as file:
        config_data = json.load(file)

    config_data['servers'] = [
        {"url": f"http://localhost:{port}{prefix}"},
        {"url": f"http://{domain}:{port}{prefix}"}
    ]

    new_config_file_path = 'static/config/config.json'

    with open(new_config_file_path, 'w') as new_file:
        json.dump(config_data, new_file, indent=2)