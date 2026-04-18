import json

nb = json.load(open('kaggle_train.ipynb'))

# Fix Cell 1 — only install packages NOT already on Kaggle
# Kaggle pre-installs: numpy, scipy, lightgbm, xgboost, catboost, torch, pandas, joblib, requests
# Installing those again causes scipy/numpy version conflicts
nb['cells'][1]['source'] = [
    "# -- 1. Install ONLY packages not pre-installed on Kaggle ---------------\n",
    "# DO NOT install numpy/scipy/lightgbm/xgboost/catboost/torch — already on Kaggle\n",
    "# Installing them again causes scipy <-> numpy version conflicts\n",
    "import subprocess, sys\n",
    "\n",
    "packages = [\n",
    "    'pandas-ta',          # not pre-installed\n",
    "    'loguru',             # not pre-installed\n",
    "    'optuna',             # not pre-installed\n",
    "    'optuna-integration[lightgbm]',  # not pre-installed\n",
    "]\n",
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + packages, check=True)\n",
    "print('Dependencies installed (used pre-installed numpy/scipy/lgbm/xgb/catboost/torch)')\n",
]

with open('kaggle_train.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Notebook fixed! Cell 1 now only installs packages missing from Kaggle.')
print('No more numpy/scipy version conflicts.')
