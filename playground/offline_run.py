import os
import sys

os.system("nohup sh -c '" +
          sys.executable + " brute_force_opt_gate.py > res.txt" +
          "' &")
