#pragma once
#include <cstdint>
#include <vector>

#include "constants.h"

void load_wav_mono_16k(const char *path, float *audio);

void compute_log_mel(const float *audio, float output[N_MELS][N_FRAMES]);