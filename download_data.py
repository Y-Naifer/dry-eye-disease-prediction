"""Download the Kaggle 'Dry Eye Disease' dataset into data/."""
import shutil
from pathlib import Path

import kagglehub

path = kagglehub.dataset_download('dakshnagra/dry-eye-disease')
Path('data').mkdir(exist_ok=True)
for f in Path(path).glob('*.csv'):
    shutil.copy(f, Path('data') / f.name)
    print('copied', f.name)
