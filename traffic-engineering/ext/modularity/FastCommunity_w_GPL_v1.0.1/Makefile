# Aaron Clauset
# Makefile  Oct2003
# feel free to hack this to pieces

#### local macros
# remove without fussing about it
RM = /bin/rm -f

# compiler name and flags
CCC = g++
CCFLAGS = -O3 -fomit-frame-pointer -funroll-loops -fforce-addr -fexpensive-optimizations -Wno-deprecated

# loader flags
LDFLAGS = 

### local program information
EXEC=FastCommunity_wMH
SOURCES= fastcommunity_w_mh.cc

### intermediate objects
OBJECTS = $(SOURCES: .cc=.o)

### includes
INCLUDES = 

### headers
HEADERS = maxheap.h vektor.h

### targets, dependencies and actions
$(EXEC): $(OBJECTS) Makefile
	$(LINK.cc) $(CCFLAGS) -o $(EXEC) $(OBJECTS)

### sort out dependencies
depend:
	makedepend $(INCLUDES) $(HEADERS) $(SOURCES)

### housekeeping

clean:
	$(RM) *.o *~ $(EXEC)

