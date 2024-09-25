#ifndef SOUNDS_H
#define SOUNDS_H

#include <stdint.h>

// standard rates, samplerate can be 44100 for higher rates

#define SAMPLERATE 22050
#define FRAME_SIZE 1024
#define HOP_SIZE 512

typedef struct {
    uint32_t soundSize;
    uint16_t *data;
} sound_t;

#endif // SOUNDS_H