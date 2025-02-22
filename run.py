import subprocess
import sys


if __name__ == "__main__":
    subprocess.run([f'{sys.executable}', "src/main.py", "--cfg-path", "configs/config.yaml", "--question", "알래스카의 역사에 대해서 알려줘."])