#!/bin/bash
gunicorn --bind 0.0.0.0:$PORT --log-level=debug webserver.app:app --daemon &
exec python ./webserver/worker.py &fg
