#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <stdint.h> 
#include <string.h>

#include "sounds.h"

 /*to implement:*/
 /*zero crossing rate*/
 /*root mean squared energy*/

int16_t* rmse(sound_t *sound, int hopSize, int frameSize) {
    int16_t *rmse = (int16_t*) malloc(sound->soundSize);

    for(int i = 0; i < sound->soundSize; i+= hopSize) {
        int16_t frameSum = 0;

        for(int z = 0; z < frameSize; z++) {
            frameSum += pow(sound->data[z],2);
        }

        frameSum /= frameSize;
        rmse[i] = sqrt(frameSum);
    }
   return rmse; 
}

/*int16_t* zcr (sound_t *sound, int hopSize, int frameSize) {*/
    /*int16_t *zcr= (int16_t*) malloc(sound->soundSize);*/

    /*for(int i = 0; i < sound->soundSize; i+= hopSize) {*/
        /*int16_t zeroSum = 0;*/
        
        /*for(int z = 0; z < frameSize-1;z++) {*/
            /*zeroSum += sound->data[z+1] - sound->data[z];*/
        /*}*/

    /*}*/
    /*zeroSum*/

/*}*/

int16_t zcr(sound_t *sound, int hopSize, int frameSize) {
    int16_t zcr = 0;
    for(int i = 0; i < sound->soundSize; i++) {
        int index = frame_start + i;

        if ((signal[index] > 0 && signal[index + 1] < 0) || 
            (signal[index] < 0 && signal[index + 1] > 0)) {
                zero_crossings++;
        }
    }
    return zcr/frameSize;
}
