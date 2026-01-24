#include "audio_transform.h"

#include <cstdint>
#include <cstddef> 
#include <fstream>
#include <algorithm>

#include "kiss_fft.h"

size_t load_wav_mono_16k(const char *path, float *audio, size_t maxSamples) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fill(audio, audio + maxSamples, 0.0f);
        return 0;
    }

    f.seekg(44); // skip WAV header

    size_t count = 0;
    int16_t s;

    // read samples
    while (count < maxSamples &&
           f.read(reinterpret_cast<char*>(&s), sizeof(int16_t)))
    {
        audio[count++] = static_cast<float>(s) / 32768.0f;
    }

    // pad remaining buffer with silence
    if (count < maxSamples) {
        std::fill(audio + count, audio + maxSamples, 0.0f);
    }

    return count; // number of samples actually read
}

void compute_log_mel(const float *audio, float output[N_MELS][N_FRAMES]) {
    kiss_fft_cfg cfg = kiss_fft_alloc(N_FFT, 0, nullptr, nullptr);

    float frame[N_FFT];
    kiss_fft_cpx fft_out[N_FFT];

    for (int t = 0; t < N_FRAMES; ++t) {
        int offset = t * HOP_LENGTH;

        for (int i = 0; i < N_FFT; ++i)
            frame[i] = audio[offset + i] * hann[i];

        kiss_fft(cfg, (kiss_fft_cpx *)frame, fft_out);

        float power[FFT_BINS];
        for (int k = 0; k < FFT_BINS; ++k)
            power[k] =
                fft_out[k].r * fft_out[k].r + fft_out[k].i * fft_out[k].i;

        for (int m = 0; m < N_MELS; ++m) {
            float sum = 0.0f;
            for (int k = 0; k < FFT_BINS; ++k)
                sum += power[k] * mel_fb[m][k];

            output[m][t] = 10.0f * log10f(fmaxf(sum, 1e-10f));
        }
    }

    free(cfg);
}