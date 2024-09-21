CC = gcc
CFLAGS = -g -Wall -Wextra -Wpedantic --std=c17

#target
readWav: readWav.o
	$(CC) $(CFLAGS) $^ -o $@

#object file
readWav.o: readWav.c
	$(CC) $(CFLAGS) -c $< -o $@

#cleanup
clean:
	rm *.o readWav

.PHONY: clean
