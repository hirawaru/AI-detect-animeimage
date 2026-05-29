#!/usr/bin/env python3
import time
import json
import os
import subprocess
import re

HISTORY = 'results/training_history.json'
# Also watch the training stderr log for live epoch output
TRAIN_ERR = os.environ.get('TRAIN_ERR_FILE', 'training_20epochs.err')
TARGET = int(os.environ.get('STOP_TARGET', '10'))
SLEEP = int(os.environ.get('WATCH_SLEEP', '10'))
KILL_SCRIPT = 'scripts/kill_src_train.ps1'
LOG = 'watch_stop10.log'


def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"{ts} - {msg}"
    print(line, flush=True)
    try:
        with open(LOG, 'a', encoding='utf-8') as fh:
            fh.write(line + '\n')
    except Exception:
        pass


log(f"Watching {HISTORY} and {TRAIN_ERR} for target={TARGET}")

prev_n = 0
prev_epoch_from_err = 0

epoch_regex = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)")

while True:
    try:
        # Check training history file first
        if os.path.exists(HISTORY):
            try:
                with open(HISTORY, 'r', encoding='utf-8') as f:
                    h = json.load(f)
                n = len(h.get('train_loss', []))
            except Exception:
                n = 0
            if n != prev_n:
                prev_n = n
                log(f"History recorded epochs: {n}")
                if n >= TARGET:
                    log(f"Target reached via history: {n} >= {TARGET}; invoking kill script")
                    try:
                        subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', KILL_SCRIPT], check=True)
                        log('Kill script executed')
                    except Exception as e:
                        log(f'Kill script failed: {type(e).__name__}: {e}')
                    break
        # Next, parse training stderr log for live epoch prints
        if os.path.exists(TRAIN_ERR):
            try:
                with open(TRAIN_ERR, 'r', encoding='utf-8', errors='ignore') as f:
                    data = f.read()
                matches = epoch_regex.findall(data)
                if matches:
                    last = matches[-1]
                    epoch_num = int(last[0])
                    total = int(last[1])
                    if epoch_num != prev_epoch_from_err:
                        prev_epoch_from_err = epoch_num
                        log(f"Parsed epoch from log: {epoch_num}/{total}")
                    if epoch_num >= TARGET:
                        log(f"Target reached via log: {epoch_num} >= {TARGET}; invoking kill script")
                        try:
                            subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', KILL_SCRIPT], check=True)
                            log('Kill script executed')
                        except Exception as e:
                            log(f'Kill script failed: {type(e).__name__}: {e}')
                        break
            except Exception as e:
                log(f"Error reading {TRAIN_ERR}: {type(e).__name__}: {e}")
    except Exception as e:
        log(f"Watcher error: {type(e).__name__}: {e}")
    time.sleep(SLEEP)

log('Watcher exiting')
