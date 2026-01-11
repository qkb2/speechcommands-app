# ExecuTorch + KissFFT SPEECHCOMMANDS C++ Application

This is a simple C++ demo application that uses the ExecuTorch library and KissFFT for SPEECHCOMMANDS CNN model inference.

## Build instructions

0. Export the model (model.ipynb) or use a PTE model already provided.

1. The ExecuTorch repository and KissFFT repository need to be downloaded to third-party directory. 
   ```bash
    mkdir third-party
    cd third-party
    git clone https://github.com/pytorch/executorch.git
    git clone https://github.com/mborgerding/kissfft.git
   ```
   Tested versions: 39c9781 for kissfft and 8c84780 for executorch.

2. Install dev requirements for ExecuTorch (venv advised):

    ```bash
    pip install -r requirements.txt
    ```

3. Build the project (python source must have executorch installed):
   ```bash
   ./build.sh
   ```

4. Run the demo application:
   ```bash
   ./build/bin/speechcommands_app cat_test.wav
   ```

## Dependencies

- CMake 3.29 or higher
- C++17 compatible compiler
- ExecuTorch library (release/1.0)
- KissFFT library

## Notes

- Make sure you have the correct model file (`.pte`) compatible with ExecuTorch.
- This app can be tested with wav files (from SPEECHCOMMANDS etc.) - it uses first second of the recording (must be sorted wav) to infer the word.
