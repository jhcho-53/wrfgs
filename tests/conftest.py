import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                      # for `import arguments`, `import scene`
sys.path.insert(0, os.path.join(ROOT, "tools"))  # for `import mw2wrfgs.*`
