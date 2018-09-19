# Directories
SRCDIR = .
OBJDIR = ../../obj
BINDIR = ../../bin

# Files
HOSTS = $(wildcard $(SRCDIR)/*.cc)
KERNELS = $(wildcard $(SRCDIR)/kernels/*.cc)
OBJS = $(patsubst $(SRCDIR)/%.cc, $(OBJDIR)/%.o, $(HOSTS)) $(patsubst $(SRCDIR)/kernels/%.cc, $(OBJDIR)/%.o, $(KERNELS))
BIN = $(BINDIR)/mf

# Common
ROCM = /opt/rocm

# Compilers
CC = $(ROCM)/bin/hipcc

# Compiler options
CCFLAGS = -std=c++11 -Wall
LDFLAGS = -std=c++11 -Wall
ifeq ($(DEBUG), 1)
  CCFLAGS += -g
  LDFLAGS += -g
else
  CCFLAGS += -O2
  LDFLAGS += -O2
endif
LIBS = -L$(CUDA)/lib64 -lm

all: make_dirs $(BIN)

make_dirs:
	mkdir -p $(OBJDIR) $(BINDIR)

$(BIN): $(OBJS)
	$(CC) -o $@ $(LDFLAGS) $^ $(LIBS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cc
	$(CC) -o $@ $(CCFLAGS) -c $<

$(OBJDIR)/%.o: $(SRCDIR)/kernels/%.cc
	$(CC) -o $@ $(CCFLAGS) -c $<

.PHONY: clean
clean:
	rm -fr $(BINDIR) $(OBJDIR)

