CC      ?= gcc
CFLAGS  ?= -O3 -Wall -Wextra -fopenmp -Iinclude
LDFLAGS ?= -fopenmp -lm

ifeq ($(OS),Windows_NT)
  SHARED_EXT := dll
  SHARED_FLAGS := -shared -DXCRUNCHER_BUILD_DLL -static
  DL_LIBS :=
  EXE := .exe
else
  UNAME_S := $(shell uname -s)
  ifeq ($(UNAME_S),Darwin)
    SHARED_EXT := dylib
    SHARED_FLAGS := -dynamiclib -fPIC
    DL_LIBS :=
  else
    SHARED_EXT := so
    SHARED_FLAGS := -shared -fPIC
    DL_LIBS := -ldl
  endif
  EXE :=
endif

SHARED_LIB := libxcruncher.$(SHARED_EXT)
ifeq ($(OS),Windows_NT)
  SHARED_LIB := xcruncher.dll
endif

.PHONY: all test bench bench-100m repro clean

all: $(SHARED_LIB) test_xcruncher$(EXE) bench_xcruncher$(EXE)

$(SHARED_LIB): src/xcruncher.c include/xcruncher.h
	$(CC) $(CFLAGS) $(SHARED_FLAGS) src/xcruncher.c -o $(SHARED_LIB) $(LDFLAGS)

test_xcruncher$(EXE): tests/test_xcruncher.c src/xcruncher.c include/xcruncher.h
	$(CC) $(CFLAGS) tests/test_xcruncher.c src/xcruncher.c -o test_xcruncher$(EXE) $(LDFLAGS)

bench_xcruncher$(EXE): bench/bench_xcruncher.c src/xcruncher.c include/xcruncher.h
	$(CC) $(CFLAGS) bench/bench_xcruncher.c src/xcruncher.c -o bench_xcruncher$(EXE) $(LDFLAGS) $(DL_LIBS)

repro_raw_limbs$(EXE): bench/repro_raw_limbs.c src/xcruncher.c include/xcruncher.h
	$(CC) $(CFLAGS) bench/repro_raw_limbs.c src/xcruncher.c -o repro_raw_limbs$(EXE) $(LDFLAGS) $(DL_LIBS)

test: $(SHARED_LIB) test_xcruncher$(EXE)
	./test_xcruncher$(EXE)
	python3 tests/test_python.py || python tests/test_python.py

bench: bench_xcruncher$(EXE)
	./bench_xcruncher$(EXE)

bench-100m: bench_xcruncher$(EXE)
	./bench_xcruncher$(EXE) --100m

repro: repro_raw_limbs$(EXE)
	./repro_raw_limbs$(EXE) --synthetic-100m --runs 7 --verify-gmp

clean:
	rm -f *.so *.dll *.dylib test_xcruncher test_xcruncher.exe bench_xcruncher bench_xcruncher.exe repro_raw_limbs repro_raw_limbs.exe python/xcruncher/*.so python/xcruncher/*.dll python/xcruncher/*.dylib
