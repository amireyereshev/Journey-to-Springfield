#!/bin/bash

mkdir -p data/raw

python3 - <<'PY'
import gdown
urls = {
    'train.csv': '1ERwQ5odiK1Zvi1LtjpkzCMUswYsAX8_K',
    'test.csv': '1fGw_-RFwvn_LEdt91Jq-7A-wzG6mmH8r',
    'submission.csv': '199Mt4OYZNaelT83U-HGDsEYs2YcUGQ6y'
}
for name, url in urls.items():
    print(f'Downloading {name}...')
    gdown.download(url, f'data/raw/{name}', quiet=False)
PY
