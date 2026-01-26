#pragma once

#define SAMPLE_RATE 16000 // 1 s
#define WIN_LENGTH 512    // 32 ms
#define HOP_LENGTH 256    // 16 ms
#define N_FFT 512         // 31.25 Hz
#define N_MELS 40         //
#define N_FRAMES 63       // depends on clip length
#define FFT_BINS 257      // 512/2 + 1
#define TIME_LENGTH 63    // (16000 - 512)/256 + 1
