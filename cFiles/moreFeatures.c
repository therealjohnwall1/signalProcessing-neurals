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

