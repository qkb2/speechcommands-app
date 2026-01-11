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
    float audio[SAMPLE_RATE];

    auto k = load_wav_mono_16k(argv[1], audio, SAMPLE_RATE);
    if (k < 1) {
        std::cerr << "File not loaded properly." << std::endl;
        return -1;
    }

    compute_log_mel(audio, input);

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
            const char* label = word_map[max_idx];
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
