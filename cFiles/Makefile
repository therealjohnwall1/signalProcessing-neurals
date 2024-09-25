CC = gcc
CFLAGS = -g -Wall -Wextra -Wpedantic --std=c17

#targets
features: readWav.o features.o
	$(CC) $(CFLAGS) $^ -o $@ -lm

#object files
readWav.o: readWav.c
	$(CC) $(CFLAGS) -c $< -o $@

features.o: features.c
	$(CC) $(CFLAGS) -c $< -o $@


#cleanup
clean:
	rm *.o features 

.PHONY: clean
