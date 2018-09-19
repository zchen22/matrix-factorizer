hip = src/hip
cuda = src/cuda
src_path =

ifneq ($(shell which hipconfig), )
    src_path += $(hip)
else
    ifneq ($(shell which nvcc), )
        src_path += $(cuda)
    endif
endif

ifeq ($(src_path), )
    $(error No HIP or CUDA found)
endif

all: $(src_path)
	make -C $<

clean: $(src_path)
	make -C $< clean

