#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <stdint.h> 
#include <string.h>

#include "sounds.h"

int readWav(char *path, sound_t *sound) {
	FILE *file;
	char magic[4];
	int32_t filesize;
	int32_t format_length;		// 16
	int16_t format_type;		// 1 = PCM
	int16_t num_channels;		// 1
	int32_t sample_rate;		// 44100
	int32_t bytes_per_second;	// sample_rate * num_channels * bits_per_sample / 8
	int16_t block_align;		// num_channels * bits_per_sample / 8
	int16_t bits_per_sample;	// 16
	int32_t data_size;

    file = fopen(path, "rb");
    assert(file != NULL);
     
	// RIFF header
    fread(magic,1,4,file);
    assert(strncmp(magic,"RIFF",4) == 0);    

    fread(&filesize, 4, 1, file);

    fread(magic,1,4,file);
    assert(strncmp(magic,"WAVE",4) == 0);    
    
    // FMT chunk 
    fread(magic,1,4,file);
    assert(strncmp(magic,"fmt ",4) == 0);    

    fread(&format_length, 4, 1, file);
    fread(&format_type, 2, 1, file);
    assert(format_type == 1);

	fread(&num_channels,2,1,file);

	// change this later idk how many channels
	assert(num_channels == 1);

    printf("hit\n");
    fread(&sample_rate, 4, 1, file);
    // STANDARD SAMPLING RATE -> follows nyquist
    /*assert(sample_rate == 44100);*/

    printf("actual sample rate: %d\n", sample_rate);

    assert(sample_rate == SAMPLERATE);
    fread(&bytes_per_second, 4, 1, file);
    fread(&block_align, 2, 1, file);
    fread(&bits_per_sample, 2, 1, file);
    assert(bits_per_sample == 16);

    fread(magic, 1, 4, file);

    assert(strncmp(magic,"data",4) == 0);

    fread(&data_size, 4, 1, file);
    sound->data = (uint16_t*) malloc(data_size);

    if(fread(sound->data, 1, data_size, file) != data_size) {
        free(sound->data);
        return 1;
    }
    sound->soundSize = data_size/2;
    return 0;
}

/*int main() {*/
    /*printf("hitter\n");*/
    /*sound_t s;*/
    /*char* path = "sampleAudio/redhot.wav";*/
    /*readWav(path,&s);*/
    /*printf("# of samples : %d \n", s.soundSize);*/
    /*return 0;*/
/*}*/
