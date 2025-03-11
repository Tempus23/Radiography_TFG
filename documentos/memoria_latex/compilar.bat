@echo off
echo Compilando con pdflatex...
pdflatex memoria_tfg

echo Ejecutando biber...
biber memoria_tfg

echo Compilando nuevamente con pdflatex...
pdflatex memoria_tfg
pdflatex memoria_tfg

echo Proceso completado.