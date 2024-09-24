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



double* hanningWindow(uint16_t* frame, int frameSize) {
    double* copyFrame = (double*)malloc(sizeof(double) * frameSize);

    for (int n = 0; n < frameSize; n++) {
        copyFrame[n] *= 0.5 * (1 - cos(2.0 * PI * n / (frameSize - 1)));
    }
    return copyFrame;
}

double* rmse(sound_t *sound, int hopSize, int frameSize, int *outputSize) {
    int n_frames = NFRAMES(sound->soundSize, hopSize, frameSize);
    *outputSize = n_frames;
    double *rmse = (double*) malloc(n_frames*sizeof(double));

    int frame_dx = 0;
    for(int i = 0; i < sound->soundSize - frameSize; i+= hopSize) {
        double frameSum = 0;
        double* windowedFrame = hanningWindow(&sound->data[i * hopSize], frameSize);
        for(int z = 0; z < frameSize; z++) {
            // frameSum += pow(sound->data[i + z], 2);
            frameSum += pow(windowedFrame[z],2);
        }
        free(windowedFrame);

        rmse[frame_dx] = sqrt(frameSum / frameSize);
        printf("val = %d\n", rmse[frame_dx]);
        frame_dx++;
    }
   return rmse; 
}

int16_t zcr(sound_t *sound, int hopSize, int frameSize) {
    int16_t zcr = 0;
    for(int i = 0; i < sound->soundSize; i++) {
        int index = frameSize+ i;

        if ((sound->data[index] > 0 && sound->data[index + 1] < 0) || 
            (sound->data[index] < 0 && sound->data[index + 1] > 0)) {
                zcr++;
        }
    }
    return zcr/frameSize;
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

/*spectral vec -> (frequency bins, frames)*/
/*use hanning windowing funciton*/
// double complex* shortTimeFT(sound_t *sound, int hopSize, int frameSize) {
// }

void print_Arr(void* arr, int size) {
    double* castArr = (double*) arr;
    for (int i = 0; i < size; i++) {
        printf("%d ", castArr[i]);
    }
    printf("\n");
}

#include "readWav.h"
int main() {
    sound_t s;
    char* path = "sampleAudio/redhot.wav";
    readWav(path,&s);
    print_Arr(&s, s.soundSize);
    /*printf("# of samples : %d \n", s.soundSize);*/
    /*int* rootMeanSize = (int*) malloc(sizeof(int)); */
    /*int16_t *rootMean = rmse(&s, HOP_SIZE, FRAME_SIZE, rootMeanSize); */
    /*printf("root mean size = %d\n", *rootMeanSize);*/
    /*print_Arr(rootMean, 100);*/

    return 0;
}










