#!/usr/bin/env python3
import requests

IMG = 'data/test/Natural/10d92uq.jpg'
URL = 'http://localhost:8000/predict'

try:
    with open(IMG, 'rb') as fh:
        files = {'file': fh}
        r = requests.post(URL, files=files, timeout=30)
    print('STATUS', r.status_code)
    print(r.text)
except Exception as e:
    print('ERROR', e)
