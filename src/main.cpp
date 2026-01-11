#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <iostream>
#include <limits>

using namespace ::executorch::extension;

int main(int argc, char *argv[]) {
    // Load the model.
    Module module("tiny_kws2.pte");

    // Create an input tensor.
    float input[1 * 40 * 63];
    auto tensor = from_blob(input, {1, 40, 63});

    // Perform an inference.
    const auto result = module.forward(tensor);

    if (result.ok()) {
        // Retrieve the output data.
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

        std::cout << "Success! Predicted class is: " << max_idx << std::endl;
    }
}
