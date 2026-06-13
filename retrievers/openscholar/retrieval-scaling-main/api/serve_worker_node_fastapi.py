import os
import json
import traceback
import datetime
import socket
import threading
import queue
import time

import hydra
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from api.api_index import get_datastore


DS_DOMAIN = os.getenv('DS_DOMAIN')
NUM_SHARDS = int(os.getenv('NUM_SHARDS'))
NUM_SHARDS_PER_WORKER = int(os.getenv('NUM_SHARDS_PER_WORKER'))
WORKER_ID = int(os.getenv('WORKER_ID'))

shard_ids = [i for i in range(WORKER_ID * NUM_SHARDS_PER_WORKER, (WORKER_ID + 1) * NUM_SHARDS_PER_WORKER)]


def load_config():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    hydra.initialize(config_path="conf")

    overrides = []
    for key, value in os.environ.items():
        if key.startswith('HYDRA_OVERRIDE_'):
            config_key = key.replace('HYDRA_OVERRIDE_', '').lower().replace('__', '.')
            overrides.append(f"{config_key}={value}")

    config_name = os.getenv('HYDRA_CONFIG_NAME', 'a100')
    cfg = hydra.compose(config_name=config_name, overrides=overrides)
    print(OmegaConf.to_yaml(cfg))
    return cfg


class SearchRequest(BaseModel):
    query: str | list
    n_docs: int
    domains: str
    additional_metadata: list = []


class Item:
    def __init__(self, query=None, query_embed=None, domains="MassiveDS", n_docs=1, additional_metadata=[]) -> None:
        self.query = query
        self.query_embed = query_embed
        self.domains = domains
        self.n_docs = n_docs
        self.additional_metadata = additional_metadata
        self.searched_results = None


class SearchQueue:
    def __init__(self, log_queries=False):
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.current_search = None
        self.cfg = load_config()
        self.cfg.datastore.domain = DS_DOMAIN
        self.cfg.datastore.embedding.num_shards = NUM_SHARDS
        self.datastore = get_datastore(self.cfg, shard_ids)

        self.log_queries = log_queries
        self.query_log = 'cached_queries.jsonl'

    def search(self, item):
        with self.lock:
            if self.current_search is None:
                self.current_search = item
                if self.log_queries:
                    now = datetime.datetime.now()
                    formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
                    with open(self.query_log, 'a+') as fin:
                        fin.write(json.dumps({'time': formatted_time, 'query': item.query}) + '\n')
                results = self.datastore.search(item.query, item.n_docs, item.additional_metadata)
                self.current_search = None
                return results
            else:
                future = threading.Event()
                self.queue.put((item, future))
                future.wait()
                return item.searched_results

    def process_queue(self):
        while True:
            item, future = self.queue.get()
            with self.lock:
                self.current_search = item
                item.searched_results = self.datastore.search(item.query, item.n_docs, item.additional_metadata)
                self.current_search = None
            future.set()
            self.queue.task_done()


app = FastAPI()
search_queue = SearchQueue()
threading.Thread(target=search_queue.process_queue, daemon=True).start()


@app.post("/search")
def search(req: SearchRequest):
    try:
        item = Item(
            query=req.query,
            domains=req.domains,
            n_docs=req.n_docs,
            additional_metadata=req.additional_metadata,
        )
        timer = threading.Timer(60.0, lambda: (_ for _ in ()).throw(TimeoutError('Search timed out after 60 seconds')))
        timer.start()
        try:
            results = search_queue.search(item)
            timer.cancel()
            return {
                "message": f"Search completed for '{item.query}' from {item.domains}",
                "query": item.query,
                "n_docs": item.n_docs,
                "results": results,
            }
        except TimeoutError as e:
            timer.cancel()
            raise HTTPException(status_code=408, detail=str(e))
    except Exception as e:
        tb_lines = traceback.format_exception(e.__class__, e, e.__traceback__)
        error_message = f"An error occurred: {str(e)}\n{''.join(tb_lines)}"
        raise HTTPException(status_code=500, detail=error_message)


@app.get("/current_search")
def current_search():
    with search_queue.lock:
        current = search_queue.current_search
        if current:
            return {
                "current_search": current.query,
                "domains": current.domains,
                "n_docs": current.n_docs,
            }
        return {"message": "No search currently in progress"}


@app.get("/queue_size")
def queue_size():
    size = search_queue.queue.qsize()
    return {"queue_size": size}


@app.get("/")
def home():
    return "Hello! What you are looking for?"


def find_free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main():
    port = 8000
    server_id = socket.gethostname()
    chunk_id = '-'.join([str(id) for id in shard_ids])
    domain_name = DS_DOMAIN
    endpoint = f'{server_id}:{port}/search'
    print(f'Running at {endpoint}')
    with open('running_ports_massiveds.jsonl', 'a+') as fout:
        info = {
            'domain_name': f'{domain_name}',
            'chunk_id': f'{chunk_id}',
            'endpoint': f'{endpoint}',
        }
        fout.write(json.dumps(info) + '\n')

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == '__main__':
    main()
