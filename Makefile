CC = gcc
CFLAGS = -g -Wall -Wextra -Wpedantic --std=c17

#targets
extractAudio: readWav.o timeDFeatures.o
	$(CC) $(CFLAGS) $^ -o $@

#object files
readWav.o: readWav.c
	$(CC) $(CFLAGS) -c $< -o $@

timeDFeatures.o: timeDFeatures.c
	$(CC) $(CFLAGS) -c $< -o $@


#cleanup
clean:
	rm *.o readWav

.PHONY: clean
