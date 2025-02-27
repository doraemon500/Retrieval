import subprocess
import sys


if __name__ == "__main__":
    subprocess.run([f'{sys.executable}', "src/main.py", "--cfg-path", "configs/config.yaml", "--question", "지미 카터"])