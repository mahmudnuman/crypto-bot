import json

nb = json.load(open('kaggle_train.ipynb'))

# Fix Cell 1 — install scipy>=1.14.0 which supports numpy 2.0.x (in-memory on Kaggle)
# Root cause: Kaggle kernel starts with numpy 2.0.2 in memory.
# System scipy on disk was compiled for numpy>=2.1.0 (needs _center).
# scipy 1.14+ was specifically redesigned to support numpy 2.0.x.
# Installing it to user site-packages takes priority over system scipy.
nb['cells'][1]['source'] = [
    "# -- 1. Install packages (scipy 1.14+ for numpy 2.0.x compatibility) ----\n",
    "import subprocess, sys\n",
    "\n",
    "# KEY FIX: Kaggle kernel has numpy 2.0.2 in memory.\n",
    "# System scipy was built for numpy 2.1+, causing _center ImportError.\n",
    "# scipy>=1.14.0 explicitly supports numpy 2.0.x.\n",
    "# Installing to user site-packages takes priority over system site-packages.\n",
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',\n",
    "    'scipy>=1.14.0',                  # supports numpy 2.0.x\n",
    "    'lightgbm',                       # reinstall linked to new scipy\n",
    "    'pandas-ta',\n",
    "    'loguru',\n",
    "    'optuna',\n",
    "    'optuna-integration[lightgbm]',\n",
    "], check=True)\n",
    "\n",
    "import numpy as np\n",
    "print(f'numpy version: {np.__version__}')\n",
    "print('Dependencies ready')\n",
]

with open('kaggle_train.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Notebook Cell 1 fixed: scipy>=1.14.0 install')
