#!/bin/bash
WS=./webserver
cd "$WS"
gunicorn --bind 0.0.0.0:$PORT --log-level=debug app:app --daemon
