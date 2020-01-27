import os

import redis
from rq import Worker, Queue, Connection

listen = ['default']

# redis_url = os.getenv('REDISTOGO_URL', 'redis://localhost:6379')
redis_url = "redis-12862.c93.us-east-1-3.ec2.cloud.redislabs.com"

# conn = redis.from_url(redis_url)
conn = redis.Redis(host=redis_url, port=12862, db="nepali-nlp", password="rtw5ZgU4fbtqJ44cyIGEQfQtMHvaHhWq")


if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(list(map(Queue, listen)))
        worker.work()
