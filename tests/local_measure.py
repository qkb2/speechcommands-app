import subprocess
import time
import os
import glob

# CONFIG
APP_PATH = "./build/bin/speechcommands_app"
DATA_DIR = "./tests/mel_bin"

bin_files = sorted(glob.glob(os.path.join(DATA_DIR, "mel_*.bin")))

total = 0
correct = 0
times = []
results = []  # (filename, expected, predicted, match, time)

for bin_path in bin_files:
    base = os.path.splitext(bin_path)[0]
    txt_path = base + ".txt"

    if not os.path.exists(txt_path):
        print(f"Missing txt file for {bin_path}, skipping")
        continue

    # Read expected value
    with open(txt_path, "r") as f:
        expected = f.read().strip()

    # Run inference
    start = time.perf_counter()
    proc = subprocess.run(
        [APP_PATH, bin_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    end = time.perf_counter()

    inference_time = end - start
    times.append(inference_time)

    predicted = proc.stdout.strip()

    match = (predicted == expected)
    if match:
        correct += 1

    total += 1
    results.append((bin_path, expected, predicted, match, inference_time))

    print(
        f"{os.path.basename(bin_path)} | "
        f"expected={expected} predicted={predicted} | "
        f"{'OK' if match else 'FAIL'} | "
        f"time={inference_time:.6f}s"
    )

# Final stats
accuracy = correct / total if total > 0 else 0.0
avg_time = sum(times) / len(times) if times else 0.0

print("\n===== SUMMARY =====")
print(f"Total tests: {total}")
print(f"Correct:     {correct}")
print(f"Accuracy:    {accuracy * 100:.2f}%")
print(f"Avg time:    {avg_time:.6f}s")