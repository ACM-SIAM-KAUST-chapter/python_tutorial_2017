SLIDES=tutorial-mpi4py
PDFLATEX=pdflatex -shell-escape --interaction=batchmode

all: slides
slides:
	${PDFLATEX} ${SLIDES}.tex
	${PDFLATEX} ${SLIDES}.tex

clean:
	${RM} *.aux *.log *.fls *.nav *.snm *.out *.toc *.vrb *.pyg
distclean: clean
	${RM} ${SLIDES}.pdf
	${RM} .sconsign.dblite
