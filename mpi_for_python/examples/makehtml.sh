#! /bin/sh

mkdir -p html
rst2html index.rst > html/index.html
for source in `ls *.py *.f90`; do
pygmentize -f html -O full,style=colorful,linenos=1 \
    -o html/$source.html $source
done

