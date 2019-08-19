# Flask Webserver

### First Steps

```sh
$ pyvenv-3.5 env
$ source env/bin/activate
$ pip install -r requirements.txt
```

### Run

Run each in a different terminal window...

```sh
# redis
$ redis-server

# worker process
$ python worker.py

# the app
$ python app.py
```