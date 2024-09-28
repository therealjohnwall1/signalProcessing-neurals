#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <stdint.h> 
#include <string.h>
#include <complex.h>
#include "sounds.h"
#include <math.h>

#define I _Complex_I
#define PI acos(-1.0)
#define NFRAMES(soundSize, hopSize, frameSize) (((soundSize) - (frameSize)) / (hopSize) + 1)
#define NFREQ_BINS(frameSize) ((frameSize) / 2 + 1)
#define N_MEL_BINS 40
#define FREQ_MIN 0
#define FREQ_MAX 22500

double* hanningWindow(uint16_t* frame, int frameSize) {
    double* copyFrame = (double*)malloc(sizeof(double) * frameSize);

    for (int n = 0; n < frameSize; n++) {
        copyFrame[n] *= 0.5 * (1 - cos(2.0 * PI * n / (frameSize - 1)));
    }
    return copyFrame;
}

/*e^{ix} ->(cartesian form) cos(x) + isin(x)
magnitude = scalar, phase = imaginary*/
double complex* fourierTransform(sound_t *sound, int nVal) {
    double complex *fourierT = (double complex*) calloc(nVal, sizeof(double complex));
    for(int f = 0; f < nVal; f++) {
        for(int i = 0; i < sound->soundSize; i++) {
            fourierT[f]+= sound->data[i] * cexp(2.0 * PI * f * i / nVal*-I);
        }
    }
    return fourierT;
}

double complex* shortTimeFourierTransform(sound_t *sound, int frameSize, int hopSize) {
    int nFrames = NFRAMES(sound->soundSize, hopSize, frameSize);
    int nFreqBins = NFREQ_BINS(frameSize);
    double complex *stft = (double complex*) calloc(nFrames * nFreqBins, sizeof(double complex));

    for (int frame = 0; frame < nFrames; frame++) {
        uint16_t *currentFrame = sound->data + frame * hopSize;
        double *windowedFrame = hanningWindow(currentFrame, frameSize);
        double complex *frameFFT = fourierTransform((sound_t*)windowedFrame, frameSize);

        for (int bin = 0; bin < nFreqBins; bin++) {
            stft[frame * nFreqBins + bin] = frameFFT[bin];
        }

        free(windowedFrame);
        free(frameFFT);
    }

    return stft;
}
double freqToMel(double freq) {
    return 2595.0 * log10(1.0 + freq / 700.0);
}

double melToFreq(double mel) {
    return 700.0 * (pow(10, mel / 2595.0) - 1);
}

double** createMelFilterbank(int nMelBins, int nFreqBins, int sampleRate, int frameSize) {
    double** melFilterbank = (double**) malloc(nMelBins * sizeof(double*));
    for (int i = 0; i < nMelBins; i++) {
        melFilterbank[i] = (double*) calloc(nFreqBins, sizeof(double));
    }

    double melMin = freqToMel(FREQ_MIN);
    double melMax = freqToMel(FREQ_MAX);

    double melBinSpacing = (melMax - melMin) / (nMelBins + 1);

    double melBinEdges[nMelBins + 2];
    for (int i = 0; i < nMelBins + 2; i++) {
        melBinEdges[i] = melToFreq(melMin + i * melBinSpacing);
    }

    int fftBinEdges[nMelBins + 2];
    for (int i = 0; i < nMelBins + 2; i++) {
        fftBinEdges[i] = (int) floor((frameSize + 1) * melBinEdges[i] / sampleRate);
    }

    // Create triangular filters
    for (int m = 1; m <= nMelBins; m++) {
        for (int k = fftBinEdges[m-1]; k < fftBinEdges[m]; k++) {
            melFilterbank[m-1][k] = (double)(k - fftBinEdges[m-1]) / (fftBinEdges[m] - fftBinEdges[m-1]);
        }
        for (int k = fftBinEdges[m]; k < fftBinEdges[m+1]; k++) {
            melFilterbank[m-1][k] = (double)(fftBinEdges[m+1] - k) / (fftBinEdges[m+1] - fftBinEdges[m]);
        }
    }

    return melFilterbank;
}

double** melSpectrogram(sound_t *sound, int frameSize, int hopSize, int sampleRate) {
    double complex* stft = shortTimeFourierTransform(sound, frameSize, hopSize);
    int nFrames = NFRAMES(sound->soundSize, hopSize, frameSize);
    int nFreqBins = NFREQ_BINS(frameSize);
    
    double** melFilterbank = createMelFilterbank(N_MEL_BINS, nFreqBins, sampleRate, frameSize);

    double** melSpectrogram = (double**) malloc(nFrames * sizeof(double*));
    for (int i = 0; i < nFrames; i++) {
        melSpectrogram[i] = (double*) calloc(N_MEL_BINS, sizeof(double));
    }
    for (int frame = 0; frame < nFrames; frame++) {
        for (int bin = 0; bin < nFreqBins; bin++) {
            double magnitude = cabs(stft[frame * nFreqBins + bin]);
            double power = magnitude * magnitude;

            for (int melBin = 0; melBin < N_MEL_BINS; melBin++) {
                melSpectrogram[frame][melBin] += power * melFilterbank[melBin][bin];
            }
        }
    }

    for (int i = 0; i < N_MEL_BINS; i++) {
        free(melFilterbank[i]);
    }
    free(melFilterbank);
    free(stft);

    return melSpectrogram;
}

// frobeius norm
void normalize(sound_t *sound) {
    double norm = 0;
    for(int i = 0; i < sound-> soundSize; i++) {
        norm += sound->data[i];
    }
    norm = sqrt(norm);

    for(int i = 0; i < sound-> soundSize; i++) {
        sound->data[i] /= norm;
    }
}

void print_uint(sound_t *s, int soundSize) {
    printf("sound size: %d", soundSize);
    for (int i = 0; i < soundSize; i++) {
        printf("%d ", s->data[i]);
    }
    printf("\n");
}

void print_complex(double complex* array, int size) {
    for (int i = 0; i < size; i++) {
        printf("(%f, %f) ", creal(array[i]), cimag(array[i]));
    }
    printf("\n");
}

#include "readWav.h"

int main() {
    sound_t s;
    char* path = "../sampleAudio/redhot.wav";
    if (readWav(path, &s) == 0) {
        printf("found wav file, reading in now\n");
        normalize(&s);
    }
    
    else {
        fprintf(stderr, "Failed to read WAV file\n");
    }

    double** melSpec = melSpectrogram(&s, FRAME_SIZE, HOP_SIZE, SAMPLERATE);
    int nFrames = NFRAMES(s.soundSize, HOP_SIZE, FRAME_SIZE);

    for (int i = 0; i < nFrames; i++) {
        for (int j = 0; j < N_MEL_BINS; j++) {
            printf("%f ", melSpec[i][j]);
        }
        printf("\n");
    }

    for (int i = 0; i < nFrames; i++) {
        free(melSpec[i]);
    }
    free(s.data);
    free(melSpec);
    return 0;
}
