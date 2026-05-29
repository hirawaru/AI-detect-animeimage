#!/usr/bin/env python3
import time
import json
import os
import yaml

HISTORY = 'results/training_history.json'
CONFIG = 'config_20epochs.yaml'
LOG = 'watch_training.log'
SLEEP = 10

prev_n = 0
# Try to read target epochs from config
target = None
try:
    if os.path.exists(CONFIG):
        with open(CONFIG, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            target = int(cfg.get('training', {}).get('num_epochs', 0))
except Exception:
    target = None

def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"{ts} - {msg}"
    print(line, flush=True)
    try:
        with open(LOG, 'a', encoding='utf-8') as fh:
            fh.write(line + '\n')
    except Exception:
        pass

log(f"Watching {HISTORY}; target={target}")

while True:
    try:
        if os.path.exists(HISTORY):
            with open(HISTORY, 'r', encoding='utf-8') as f:
                h = json.load(f)
            n = len(h.get('train_loss', []))
            if n > prev_n:
                log(f"Epochs recorded: {n}")
                prev_n = n
                if target and n >= target:
                    log(f"Target reached: {n} >= {target}")
                    break
        else:
            log(f"History file not found: {HISTORY}")
    except Exception as e:
        log(f"Error reading history: {type(e).__name__}: {e}")
    time.sleep(SLEEP)

log('Watcher exiting')
