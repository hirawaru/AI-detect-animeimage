#!/usr/bin/env python3
"""
Monitor training history until a target number of epochs is reached,
then restart the web service and run a quick inference smoke test.

Usage: python scripts/monitor_epochs_and_deploy.py
"""
import time
import json
import os
import subprocess
import sys

HISTORY = 'results/training_history.json'
TARGET = 5
SLEEP = 15

def wait_for_epochs(target=TARGET):
    print(f"Waiting until at least {target} train epochs are recorded in {HISTORY}...")
    while True:
        if os.path.exists(HISTORY):
            try:
                with open(HISTORY, 'r') as f:
                    h = json.load(f)
                n = len(h.get('train_loss', []))
                print(f"Found {n} epochs recorded")
                if n >= target:
                    return n
            except Exception as e:
                print('Read error:', e)
        else:
            print('History file not found yet')
        time.sleep(SLEEP)

def restart_web():
    print('Restarting web service via docker compose...')
    try:
        subprocess.run(['docker','compose','restart','web'], check=True)
        print('Restart issued')
    except Exception as e:
        print('Failed to restart web:', e)

def smoke_test():
    print('Running smoke test against /predict')
    # find a sample image
    sample = None
    for root, dirs, files in os.walk('data/test'):
        for f in files:
            if f.lower().endswith(('.jpg','.jpeg','.png')):
                sample = os.path.join(root, f)
                break
        if sample:
            break
    if not sample:
        print('No sample image found under data/test to run smoke test')
        return
    print('Using sample', sample)
    try:
        import requests
        with open(sample, 'rb') as fh:
            r = requests.post('http://127.0.0.1:8000/predict', files={'file': fh}, timeout=30)
        print('Status', r.status_code)
        try:
            print('JSON:', r.json())
        except Exception:
            print('Response text:', r.text)
    except Exception as e:
        print('Smoke test failed:', e)

def main():
    n = wait_for_epochs(TARGET)
    print(f'Target reached: {n} epochs recorded')
    restart_web()
    # allow a few seconds for app to start
    time.sleep(5)
    smoke_test()

if __name__ == '__main__':
    main()
