"""Download the attributed UCI source and convert its worksheet to CSV."""
from pathlib import Path
import io
import urllib.request
import zipfile
import pandas as pd


def main():
    out=Path('data/uci');out.mkdir(parents=True,exist_ok=True)
    archive=out/'source.zip'
    if not archive.exists():
        url='https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip'
        with urllib.request.urlopen(url,timeout=60) as response:
            payload=response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            z.testzip()
        archive.write_bytes(payload)
    with zipfile.ZipFile(archive) as z:
        name=next(n for n in z.namelist() if n.endswith('.xls'))
        with z.open(name) as f: df=pd.read_excel(f,header=1)
    if df.shape!=(30000,25):
        raise ValueError('Unexpected source schema; inspect before continuing')
    df.to_csv(out/'credit_card_clients.csv',index=False)
    print('Converted UCI data; see data/uci/README.md for citation and license.')

if __name__=='__main__': main()
