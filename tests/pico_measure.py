import serial
import time
import os
import glob

# CONFIG
SERIAL_PORT = "/dev/ttyACM0"  
BAUDRATE = 115200  
DATA_DIR = "tests/mel_bin"
SERIAL_TIMEOUT = 2.0  # seconds

# Open serial connection
ser = serial.Serial(port=SERIAL_PORT, baudrate=BAUDRATE, timeout=SERIAL_TIMEOUT)

# Give Pico time to reset after USB connect
time.sleep(2)
ser.reset_input_buffer()

bin_files = sorted(glob.glob(os.path.join(DATA_DIR, "mel_*.bin")))

total = 0
correct = 0
times = []
results = []

for bin_path in bin_files:
    base = os.path.splitext(bin_path)[0]
    txt_path = base + ".txt"

    if not os.path.exists(txt_path):
        print(f"Missing txt file for {bin_path}, skipping")
        continue

    # Expected result
    with open(txt_path, "r") as f:
        expected = f.read().strip()

    # Read binary data
    with open(bin_path, "rb") as f:
        bin_data = f.read()

    # Send data + measure inference time
    ser.reset_input_buffer()
    start = time.perf_counter()

    ser.write(bin_data)
    ser.flush()

    # Read Pico response (one line)
    response = ser.readline()

    end = time.perf_counter()
    inference_time = end - start

    if not response:
        print(f"{os.path.basename(bin_path)} | TIMEOUT")
        continue

    predicted = response.decode("utf-8", errors="ignore").strip()

    match = predicted == expected
    if match:
        correct += 1

    total += 1
    times.append(inference_time)

    print(
        f"{os.path.basename(bin_path)} | "
        f"expected={expected} predicted={predicted} | "
        f"{'OK' if match else 'FAIL'} | "
        f"time={inference_time:.6f}s"
    )

# Close serial
ser.close()

# Final stats
accuracy = correct / total if total > 0 else 0.0
avg_time = sum(times) / len(times) if times else 0.0

print("\n===== SUMMARY =====")
print(f"Total tests: {total}")
print(f"Correct:     {correct}")
print(f"Accuracy:    {accuracy * 100:.2f}%")
print(f"Avg time:    {avg_time:.6f}s")
