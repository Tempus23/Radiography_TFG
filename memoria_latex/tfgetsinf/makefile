# Makefile para compilar memoria_tfg.tex con BibLaTeX y Biber

# Nombres de archivos
DOCUMENT = memoria_tfg
BIBFILE = bib.bib

# Programas
LATEX = pdflatex
BIBER = biber

# Opciones
LATEXOPTS = -interaction=nonstopmode -synctex=1 -file-line-error
BIBEROPTS =

# Colores para mensajes
GREEN = \033[0;32m
YELLOW = \033[0;33m
RED = \033[0;31m
NC = \033[0m # No Color

# Regla por defecto: compilar el documento completo
all: $(DOCUMENT).pdf

# Compilación completa (LaTeX + Biber + LaTeX + LaTeX)
$(DOCUMENT).pdf: $(DOCUMENT).tex $(BIBFILE)
    @echo "$(GREEN)Primera compilación LaTeX...$(NC)"
    $(LATEX) $(LATEXOPTS) $(DOCUMENT)
    @echo "$(GREEN)Ejecutando Biber...$(NC)"
    $(BIBER) $(DOCUMENT)
    @echo "$(GREEN)Segunda compilación LaTeX...$(NC)"
    $(LATEX) $(LATEXOPTS) $(DOCUMENT)
    @echo "$(GREEN)Tercera compilación LaTeX (referencias cruzadas y citas)...$(NC)"
    $(LATEX) $(LATEXOPTS) $(DOCUMENT)
    @echo "$(GREEN)¡Documento compilado con éxito!$(NC)"

# Compilación rápida (solo LaTeX, sin bibliografía)
quick: $(DOCUMENT).tex
    @echo "$(YELLOW)Compilación rápida (sin bibliografía)...$(NC)"
    $(LATEX) $(LATEXOPTS) $(DOCUMENT)
    @echo "$(YELLOW)Compilado rápidamente.$(NC)"

# Procesar bibliografía (Biber)
biber: $(DOCUMENT).bcf
    @echo "$(GREEN)Ejecutando Biber...$(NC)"
    $(BIBER) $(DOCUMENT)
    @echo "$(GREEN)Bibliografía procesada.$(NC)"

# Compilar tras cambios en la bibliografía
bibupdate: biber
    @echo "$(GREEN)Actualizando el documento con la nueva bibliografía...$(NC)"
    $(LATEX) $(LATEXOPTS) $(DOCUMENT)
    $(LATEX) $(LATEXOPTS) $(DOCUMENT)
    @echo "$(GREEN)Bibliografía actualizada.$(NC)"

# Limpiar archivos temporales
clean:
    @echo "$(RED)Eliminando archivos temporales...$(NC)"
    rm -f $(DOCUMENT).aux $(DOCUMENT).bbl $(DOCUMENT).bcf $(DOCUMENT).blg \
        $(DOCUMENT).log $(DOCUMENT).out $(DOCUMENT).run.xml $(DOCUMENT).synctex.gz \
        $(DOCUMENT).toc $(DOCUMENT).lof $(DOCUMENT).lot $(DOCUMENT).idx \
        $(DOCUMENT).ilg $(DOCUMENT).ind $(DOCUMENT).fls $(DOCUMENT).fdb_latexmk
    @echo "$(RED)Archivos temporales eliminados.$(NC)"

# Limpiar todo (incluido el PDF)
cleanall: clean
    @echo "$(RED)Eliminando el PDF...$(NC)"
    rm -f $(DOCUMENT).pdf
    @echo "$(RED)Todo limpio.$(NC)"

# Forzar recompilación completa
rebuild: cleanall all

# Visualizar el PDF
view: $(DOCUMENT).pdf
    @echo "$(GREEN)Abriendo el PDF...$(NC)"
    start $(DOCUMENT).pdf

# Ayuda
help:
    @echo "$(YELLOW)Opciones disponibles:$(NC)"
    @echo "  $(GREEN)make$(NC) - Compilar el documento completo"
    @echo "  $(GREEN)make quick$(NC) - Compilación rápida (sin bibliografía)"
    @echo "  $(GREEN)make biber$(NC) - Procesar solo la bibliografía"
    @echo "  $(GREEN)make bibupdate$(NC) - Actualizar el documento tras cambios en bibliografía"
    @echo "  $(GREEN)make clean$(NC) - Eliminar archivos temporales"
    @echo "  $(GREEN)make cleanall$(NC) - Eliminar archivos temporales y PDF"
    @echo "  $(GREEN)make rebuild$(NC) - Forzar recompilación completa"
    @echo "  $(GREEN)make view$(NC) - Abrir el PDF generado"
    @echo "  $(GREEN)make help$(NC) - Mostrar esta ayuda"

# Declarar los objetivos que no son archivos
.PHONY: all quick biber bibupdate clean cleanall rebuild view help