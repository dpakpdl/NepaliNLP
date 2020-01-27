web: gunicorn --bind 0.0.0.0:$PORT --log-level=debug app:app --daemon
worker: python ./webserver/worker.py
