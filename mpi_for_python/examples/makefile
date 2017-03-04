mpiexec=mpiexec
python=python

.PHONY:test
test: all clean

.PHONY:all
all: helloworld comm p2p coll compute_pi mandelbrot

.PHONY:helloworld
helloworld:
	mpicc helloworld.c
	mpiexec -n 5 ./a.out
	rm ./a.out && echo
	mpicxx helloworld.cxx
	mpiexec -n 5 ./a.out
	rm ./a.out && echo
	mpif90 helloworld.f90
	mpiexec -n 5 ./a.out
	rm ./a.out && echo
	${mpiexec} -n 5 ${python} helloworld.py
	@echo

.PHONY:comm
comm:
	${mpiexec} -n 5 ${python} comm_create.py
	${mpiexec} -n 7 ${python} comm_split.py

.PHONY:p2p
p2p:
	${mpiexec} -n 2 ${python} p2p_pingpong.py
	${mpiexec} -n 2 ${python} p2p_pingping.py
	${mpiexec} -n 7 ${python} p2p_exchange.py

.PHONY:coll
coll:
	${mpiexec} -n 3 ${python} coll_bcast.py
	${mpiexec} -n 4 ${python} coll_scatter.py
	${mpiexec} -n 4 ${python} coll_gather.py
	${mpiexec} -n 4 ${python} coll_allgather.py
	${mpiexec} -n 4 ${python} coll_reduce.py
	${mpiexec} -n 4 ${python} coll_allreduce.py

NP=4

.PHONY:compute_pi
compute_pi:
	${python} compute_pi-seq.py
	${mpiexec} -n ${NP} ${python} compute_pi-mpi.py
	${mpiexec} -n ${NP} ${python} compute_pi-mpi-numpy.py
	echo 10 | ${mpiexec} -n ${NP} ${python} compute_pi-mpi-stdin.py

	${mpiexec} -n 1 ${python} compute_pi-parent.py

	${python} compute_pi-montecarlo-seq.py
	${mpiexec} -n ${NP} ${python} compute_pi-montecarlo-mpi.py

.PHONY:mandelbrot
mandelbrot: mandelbrot-worker.exe
	${python} mandelbrot-seq.py
	${mpiexec} -n ${NP} ${python} mandelbrot-mpi-block.py
	${mpiexec} -n ${NP} ${python} mandelbrot-mpi-cyclic.py
	${mpiexec} -n 1 ${python} mandelbrot-master.py ${NP}
mandelbrot-worker.exe: mandelbrot-worker.f90
	mpif90 -o $@ $<


.PHONY:clean
clean:
	${RM} *.py[co]
	${RM} -r __pycache__
	${RM} mandelbrot-worker.exe


html:
	./makehtml.sh
distclean: clean
	${RM} -r html
