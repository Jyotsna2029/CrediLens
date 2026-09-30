"""Run on the machine where the original joblib files load successfully."""
import argparse
import json
from pathlib import Path
from .persistence import export_bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', required=True)
    args=parser.parse_args()
    directory=Path(args.artifacts)
    manifest=json.loads((directory/'manifest.json').read_text())
    for name in manifest['models']:
        if (directory/f'{name}.portable.joblib').exists():
            print(f'{name}: existing native export retained')
            continue
        path=export_bundle(directory,name)
        if path: print(f'{name}: native model exported to {path}')
    print('Other estimators/preprocessing still require compatible scikit-learn and dependency versions.')

if __name__=='__main__': main()
