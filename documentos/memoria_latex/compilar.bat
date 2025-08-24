@echo off

echo Compilando con pdflatex...
pdflatex memoria_tfg
IF ERRORLEVEL 1 (
    echo Error: Fallo al compilar con pdflatex.
    exit /b 1
)

echo Ejecutando biber...
biber memoria_tfg
IF ERRORLEVEL 1 (
    echo Error: Fallo al ejecutar biber.
    exit /b 1
)

echo Compilando nuevamente con pdflatex...
pdflatex memoria_tfg
IF ERRORLEVEL 1 (
    echo Error: Fallo al compilar con pdflatex (segunda pasada).
    exit /b 1
)
pdflatex memoria_tfg
IF ERRORLEVEL 1 (
    echo Error: Fallo al compilar con pdflatex (tercera pasada).
    exit /b 1
)

cd ..
cd ..
echo Proceso completado.