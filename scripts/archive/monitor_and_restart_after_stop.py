#!/usr/bin/env python3
import time
import subprocess
import os

WATCHER_LOG = 'watch_resume_stop10.log'
SMOKE_IMAGE = 'data/test/Natural/10d92uq.jpg'
DOCKER_SERVICE = ['docker', 'compose', 'restart', 'web']
LOG = 'monitor_and_restart.log'


def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"{ts} - {msg}"
    print(line, flush=True)
    try:
        with open(LOG, 'a', encoding='utf-8') as fh:
            fh.write(line + '\n')
    except Exception:
        pass


log(f"Starting monitor: watch {WATCHER_LOG}")
seen = False
while True:
    try:
        if os.path.exists(WATCHER_LOG):
            with open(WATCHER_LOG, 'r', encoding='utf-8', errors='ignore') as f:
                data = f.read()
            if 'Kill script executed' in data or 'Target reached via log' in data or 'Target reached via history' in data:
                log('Detected training stop signal in watcher log')
                seen = True
                break
    except Exception as e:
        log(f'Error reading watcher log: {e}')
    time.sleep(5)

if not seen:
    log('Exiting monitor: no stop detected')
    raise SystemExit(1)

# Restart docker web
try:
    log('Restarting Docker web service')
    subprocess.run(DOCKER_SERVICE, check=True)
    log('Docker restart command executed')
except Exception as e:
    log(f'Docker restart failed: {e}')

# Wait a bit for the container to come up
time.sleep(8)

# Run smoke-test
try:
    import requests
    url = 'http://localhost:8000/predict'
    log(f'Running smoke-test POST to {url} with image {SMOKE_IMAGE}')
    with open(SMOKE_IMAGE, 'rb') as fh:
        files = {'file': fh}
        r = requests.post(url, files=files, timeout=15)
    log(f'Smoke-test status: {r.status_code}, response: {r.text}')
except Exception as e:
    log(f'Smoke-test failed: {e}')

log('Monitor finished')
