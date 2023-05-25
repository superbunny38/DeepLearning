import subprocess
import argparse
import os
parser = argparse.ArgumentParser(description='Argparse')
parser.add_argument('--device', type=int, default = 5,
                    help='visible device')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.device}"

filepaths = ["./train_fold0.py",
    "./train_fold1.py",
    "./train_fold2.py",
    "./train_fold3.py",
    "./train_fold4.py"
]

for idx,filepath in enumerate(filepaths):
    print(f"fold{idx} training...")
    subprocess.call(["python", filepath])