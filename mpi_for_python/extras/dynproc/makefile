targets = cpi-worker-c cpi-worker-f
all: $(targets)

cpi-worker-c: cpi-worker.c
	mpicc $< -o $@

cpi-worker-f: cpi-worker.f90
	mpifort $< -o $@

clean:
	$(RM) $(targets)

test-all: all
	mpiexec python test.py ./cpi-worker-c
	mpiexec python test.py ./cpi-worker-f
	mpiexec python test.py ./cpi-worker.py
