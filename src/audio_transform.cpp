#include "audio_transform.h"

#include "kissfft/kiss_fft.h"

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