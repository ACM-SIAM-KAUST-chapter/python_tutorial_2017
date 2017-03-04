#!/bin/sh
if [ x$1 = x-n ]; then np="$1 $2"; shift; shift; fi
mpiexec $np xterm -e ipython --no-banner -i $@
