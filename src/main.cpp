#include "pico/stdio_usb.h"
#include "pico/stdlib.h"

#include <cmath>
#include <cstdint>
#include <cstdio>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

// #include "audio_transform.h"
#include "constants.h"
#include "model_pte.h"
#include "words.h"

#define INPUT_FLOATS (N_MELS * N_FRAMES)
#define INPUT_BYTES (INPUT_FLOATS * sizeof(float))

using namespace executorch::runtime;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;

// Allocate memory for ExecuTorch runtime
static uint8_t method_allocator_pool[180 * 1024];
static uint8_t activation_pool[180 * 1024];
static float input_mel[N_MELS][N_FRAMES];

bool read_exact_bytes(uint8_t *dst, size_t len) {
    size_t received = 0;
    while (received < len) {
        int c = getchar_timeout_us(5 * 1000 * 1000);
        if (c == PICO_ERROR_TIMEOUT) {
            return false;
        }
        dst[received++] = (uint8_t)c;
    }
    return true;
}

bool receive_mel_over_usb(float mel[N_MELS][N_FRAMES]) {
    // printf("Waiting for %d bytes...\n", INPUT_BYTES);
    fflush(stdout);

    return read_exact_bytes(reinterpret_cast<uint8_t *>(mel), INPUT_BYTES);
}

float run_inference(Method &method, float input_data[N_MELS][N_FRAMES]) {
    TensorImpl::SizesType input_sizes[3] = {1, N_MELS, N_FRAMES};
    TensorImpl::DimOrderType dim_order[3] = {0, 1, 2};

    TensorImpl input_impl(ScalarType::Float, 3, input_sizes, input_data,
                          dim_order);
    Tensor input_tensor(&input_impl);

    if (method.set_input(input_tensor, 0) != Error::Ok) {
        printf("ERROR: set_input failed\n");
        return -1.0f;
    }

    if (method.execute() != Error::Ok) {
        printf("ERROR: execute failed\n");
        return -1.0f;
    }

    auto output = method.get_output(0);
    if (!output.isTensor()) {
        printf("ERROR: output not tensor\n");
        return -1.0f;
    }

    const float *out_data = output.toTensor().const_data_ptr<float>();
    int numel = output.toTensor().numel();

    // Return index of max value
    int max_idx = 0;
    float max_val = out_data[0];
    for (int i = 1; i < numel; i++) {
        if (out_data[i] > max_val) {
            max_val = out_data[i];
            max_idx = i;
        }
    }
    return max_idx; // return predicted class index
}

int main() {
    stdio_init_all();
    sleep_ms(2000); // wait for USB

    runtime_init();

    MemoryAllocator method_allocator(sizeof(method_allocator_pool),
                                     method_allocator_pool);
    Span<uint8_t> planned_buffers[1] = {
        {activation_pool, sizeof(activation_pool)}};
    HierarchicalAllocator planned_memory({planned_buffers, 1});
    MemoryManager memory_manager(&method_allocator, &planned_memory);

    // Load model
    executorch::extension::BufferDataLoader loader(model_pte, model_pte_len);
    auto program_result = Program::load(&loader);
    if (!program_result.ok()) {
        printf("ERROR: Program load failed (%d)\n",
               (int)program_result.error());
        return -1;
    }
    Program program = std::move(*program_result);

    auto method_name = program.get_method_name(0);
    if (!method_name.ok()) {
        printf("ERROR: get_method_name failed\n");
        return -1;
    }

    auto method_result = program.load_method(*method_name, &memory_manager);
    if (!method_result.ok()) {
        printf("ERROR: load_method failed (%d)\n", (int)method_result.error());
        return -1;
    }
    Method method = std::move(*method_result);

    // Run inference
    while (true) {
        if (!receive_mel_over_usb(input_mel)) {
            printf("ERROR: USB receive failed\n");
            continue;
        }

        int predicted_idx = run_inference(method, input_mel);

        if (predicted_idx >= 0 && predicted_idx < word_map.size()) {
            printf("%d\n", predicted_idx);
        } else {
            printf("ERROR: Prediction out of range\n");
        }

        fflush(stdout);
    }

    return 0;
}
