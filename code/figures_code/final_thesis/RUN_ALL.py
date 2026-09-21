import os
import subprocess

script_dir = os.path.dirname(os.path.abspath(__file__))

files = [
    f
    for f in os.listdir(script_dir)
    if f.startswith("Generate") and f.endswith(".py")
]

for file in files:
    print(f"Run: {file}...")
    file_path = os.path.join(script_dir, file)
    subprocess.run(["python", file_path], check=True)