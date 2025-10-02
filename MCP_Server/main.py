# import redis
from enum import Enum, auto
from mcp.server.fastmcp import FastMCP
from load_data import perf_data_loader
mcp = FastMCP("FirstStep")

class StorageType(Enum):
    Local = auto()
    BeeGFS = auto()


class Storage:
    def __init__(self, storage_type=StorageType, description=""):
        self.type = storage_type
        self.description = description
        self.workload = 0

    def increase_workload(self):
        # example
        self.workload += 1

    def __str__(self):
        return f"Storage(type={self.type.name}, description='{self.description}', workload={self.workload})"


class Storages:
    def __init__(self):
        self.storage_list = []

    def insert_storage(self, item: Storage):
        self.storage_list.append(item)

    def get_best_storage(self):
        # REPLACED BY OTHER METHOD
        return self.storage_list[0].type.name
    
    def get_all_storages(self):
        return self.storage_list

# def init():
    # redis_client = redis.Redis("127.0.0.1", port=6379, db=0)
    # assert redis_client.ping()
@mcp.tool()
def get_benchmark():
    return perf_data_loader()
    ...

@mcp.tool()
def list_avaliable_storages():
    return storages.get_all_storages()
    # return ["SSD1","SSD2","BeeGFS"]


@mcp.tool()
def get_recomanded():
    return storages.get_best_storage()

@mcp.tool()
def hello_world():
    return "Hello World!"

storages = Storages()
storages.insert_storage(Storage(StorageType.Local,"This is a SSD drive."))
storages.insert_storage(Storage(StorageType.BeeGFS, "This is a BeeGFS clusters."))
# init()

if __name__ == "__main__":
    ...
    mcp.run(transport="stdio")
