import json

nb = json.load(open('kaggle_train.ipynb'))

# Fix Cell 1 — pin numpy>=2.1.0 alongside pandas-ta to prevent downgrade
# Root cause: pandas-ta accepts numpy>=1.17 (no upper bound), so pip may install
# numpy 2.0.x which is missing _center in numpy._core.umath that scipy needs.
# scipy on Kaggle was compiled for numpy>=2.1.0.
nb['cells'][1]['source'] = [
    "# -- 1. Install missing packages, keeping numpy>=2.1.0 for scipy compat --\n",
    "import subprocess, sys\n",
    "\n",
    "# Pin numpy>=2.1.0 FIRST — pandas-ta would otherwise drag it down to 2.0.x\n",
    "# which breaks scipy (scipy needs numpy._core.umath._center, added in 2.1.0)\n",
    "subprocess.run([\n",
    "    sys.executable, '-m', 'pip', 'install', '-q',\n",
    "    'numpy>=2.1.0',          # must come first to prevent downgrade\n",
    "    'pandas-ta',\n",
    "    'loguru',\n",
    "    'optuna',\n",
    "    'optuna-integration[lightgbm]',\n",
    "], check=True)\n",
    "\n",
    "import numpy as np\n",
    "print(f'numpy version: {np.__version__}')  # confirm >= 2.1.0\n",
    "print('Dependencies ready')\n",
]

with open('kaggle_train.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Fixed! Cell 1 now pins numpy>=2.1.0 to prevent scipy _center ImportError.')
