#include "audio_transform.h"
#include "words.h"

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

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

    for (int i = 0; i < N_MELS; i++) {
        for (int j = 0; j < N_FRAMES; j++) {
            printf("%f ", input[i][j]);
        }
    }

    auto tensor = from_blob(input, {1, N_MELS, N_FRAMES});

    const auto result = module.forward(tensor);

    if (result.ok()) {
        const auto output_tensor = result->at(0).toTensor();
        const int numel = output_tensor.numel(); // Should be 37
        const auto output = output_tensor.const_data_ptr<float>();

        int max_idx = 0;
        float max_val = std::numeric_limits<float>::lowest();
        for (int i = 0; i < numel; i++) {
            if (output[i] > max_val) {
                max_val = output[i];
                max_idx = i;
            }
        }

        if (max_idx < word_map.size()) {
            const char *label = word_map[max_idx];
            std::cout << "Success! Predicted class is: " << label << std::endl;
        } else {
            std::cerr << "Index of bounds." << std::endl;
            return -1;
        }
    } else {
        std::cerr << "Results not ok." << std::endl;
        return -1;
    }
}
