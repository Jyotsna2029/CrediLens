"""Start CrediLens with the trained model directories available on this machine."""
from pathlib import Path
import os
import subprocess
import sys


def main():
    root=Path(__file__).resolve().parent
    os.chdir(root)
    for variable, local, fallback in [
        ('CREDIT_RISK_ARTIFACTS','artifacts_windows','artifacts'),
        ('UCI_ARTIFACTS','artifacts_uci_windows','artifacts_uci')]:
        if variable not in os.environ:
            os.environ[variable]=str(root/(local if (root/local/'manifest.json').exists() else fallback))
        directory=Path(os.environ[variable]).resolve()
        print(f'{variable}: {directory}',flush=True)
        if (directory/'manifest.json').exists():
            subprocess.run([sys.executable,'-m','credit_risk.export_models','--artifacts',str(directory)],check=True)
    subprocess.run([sys.executable,'-m','uvicorn','main:app','--reload'],check=True)

if __name__=='__main__':
    try: main()
    except KeyboardInterrupt: pass
