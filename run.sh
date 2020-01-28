#!/bin/bash
exec gunicorn --bind 0.0.0.0:$PORT --log-level=debug --chdir webserver app:app
exec python ./webserver/worker.py
