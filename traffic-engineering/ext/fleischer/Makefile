# remove without fussing about it
RM = rm -f

# compiler name and flags
CC = g++
CCFLAGS = -O3 -std=c++1z -fomit-frame-pointer -funroll-loops -fforce-addr -fexpensive-optimizations -Wno-deprecated

# loader flags
LDFLAGS = 

### local program information
EXEC = fl
SOURCES = fleischer.cpp pqueue.cpp

### intermediate objects
OBJECTS = $(SOURCES:.cpp=.o)

### includes
INCLUDES =

### headers
HEADERS = pqueue.h

.PHONY: clean

all: $(EXEC)

### targets, dependencies and actions
$(EXEC): $(OBJECTS)
	$(CC) $(CCFLAGS) $(OBJECTS) -o $(EXEC) 

.cpp.o:
	$(CC) $(CCFLAGS) -c $<  -o $@

### housekeeping
clean:
	$(RM) *.o *~ $(EXEC)

