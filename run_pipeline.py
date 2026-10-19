"""Execute the complete batch pipeline."""
import subprocess
import sys
from pathlib import Path
for name in ('data_cleaning','analytics','sql_analysis','train_model'):
    print(f'\n== {name} ==',flush=True)
    subprocess.run([sys.executable,'-m',f'src.{name}'],check=True,cwd=Path(__file__).resolve().parent)
