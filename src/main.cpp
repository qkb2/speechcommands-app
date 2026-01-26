#include "constants.h"
#include "words.h"

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <chrono>
#include <iostream>
#include <limits>

using namespace ::executorch::extension;

int main(int argc, char *argv[]) {
    Module module("tiny_kws2.pte");

    float input[N_MELS][N_FRAMES];

    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        std::cerr << "Failed to open input file." << std::endl;
        return -1;
    }

    size_t read = fread(input, sizeof(float), N_MELS * N_FRAMES, f);
    fclose(f);

    if (read != N_MELS * N_FRAMES) {
        std::cerr << "Input size mismatch." << std::endl;
        return -1;
    }

    auto tensor = from_blob(input, {1, N_MELS, N_FRAMES});

    // Warm-up inference (not timed)
    auto warmup = module.forward(tensor);
    if (!warmup.ok()) {
        std::cerr << "Warm-up inference failed." << std::endl;
        return -1;
    }

    // Timed inference loop
    constexpr int kNumRuns = 1000;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kNumRuns; i++) {
        auto result = module.forward(tensor);
        if (!result.ok()) {
            std::cerr << "Inference failed at run " << i << std::endl;
            return -1;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> total_ms = end - start;
    double avg_ms = total_ms.count() / kNumRuns;

    std::cout << "Average inference time: " << avg_ms << " ms (" << kNumRuns
              << " runs)" << std::endl;


    return 0;
}