# Variables
DOCUMENT = memoria_tfg
LATEX = pdflatex
BIBER = biber

# Default target
all: $(DOCUMENT).pdf

# Main target - the PDF
$(DOCUMENT).pdf: $(DOCUMENT).tex
	@echo "Compilando con pdflatex..."
	$(LATEX) $(DOCUMENT)
	@echo "Ejecutando biber..."
	$(BIBER) $(DOCUMENT)
	@echo "Compilando nuevamente con pdflatex..."
	$(LATEX) $(DOCUMENT)
	$(LATEX) $(DOCUMENT)
	@echo "Proceso completado."

# Clean target to remove temporary files
clean:
	@echo "Limpiando archivos temporales..."
	rm -f $(DOCUMENT).aux $(DOCUMENT).bbl $(DOCUMENT).blg $(DOCUMENT).log $(DOCUMENT).out $(DOCUMENT).toc $(DOCUMENT).lof $(DOCUMENT).lot $(DOCUMENT).bcf $(DOCUMENT).run.xml
	@echo "Limpieza completada."

# Very clean target to remove temporary files AND the PDF
cleanall: clean
	@echo "Eliminando PDF..."
	rm -f $(DOCUMENT).pdf
	@echo "Limpieza completa terminada."

# Help target
help:
	@echo "Targets disponibles:"
	@echo "  all (default): Compila el documento $(DOCUMENT).tex completamente"
	@echo "  clean: Elimina archivos temporales generados durante la compilación"
	@echo "  cleanall: Elimina archivos temporales y el PDF final"
	@echo "  help: Muestra este mensaje de ayuda"

# Phony targets
.PHONY: all clean cleanall help
