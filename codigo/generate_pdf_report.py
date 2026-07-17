from pathlib import Path
import nbformat
import subprocess
import sys


def main():
    repo_root = Path(__file__).resolve().parents[1]
    nb_in = repo_root / 'resultados' / 'T2_EDA_Concreto_Grupo03_ejecutado.ipynb'
    if not nb_in.exists():
        print('No se encontró el notebook de entrada en:', nb_in)
        sys.exit(1)

    images_dir = repo_root / 'resultados' / 'imagenes'

    # Mapeo: encabezado de sección -> lista de nombres de fichero de imagen (relativos a resultados/)
    mapping = {
        '## 5. Distribuciones': ['distribucion_variables.png'],
        '## 7. Correlación y relaciones bivariadas': [
            'matriz_correlacion.png',
            'resistencia_vs_variables.png'
        ],
        '## 11. Benchmark preliminar reproducible': [
            'comparacion_modelos_r2.png',
            'comparacion_modelos_rmse.png'
        ]
    }

    nb = nbformat.read(str(nb_in), as_version=4)

    # Insertar celdas markdown con las imágenes después de la celda con el encabezado
    new_cells = []
    i = 0
    while i < len(nb.cells):
        cell = nb.cells[i]
        new_cells.append(cell)
        if cell.cell_type == 'markdown':
            text = ''.join(cell.get('source', []))
            for header, imgs in mapping.items():
                if header in text:
                    for img in imgs:
                        img_path = images_dir / img
                        rel_path = Path('imagenes') / img
                        if img_path.exists():
                            md = f'![]({rel_path.as_posix()})'
                        else:
                            md = f'> **Aviso:** imagen no encontrada: {img_path.name}'
                        insert_cell = nbformat.v4.new_markdown_cell(source=md)
                        insert_cell.metadata = {'language': 'markdown'}
                        new_cells.append(insert_cell)
        i += 1

    nb.cells = new_cells

    out_nb = repo_root / 'resultados' / 'T2_EDA_Concreto_Grupo03_con_imagenes.ipynb'
    nbformat.write(nb, str(out_nb))
    print('Notebook con imágenes creado en:', out_nb)

    # Exportar a PDF usando nbconvert (requiere jupyter + LaTeX en el sistema)
    report_dir = repo_root / 'resultados' / 'reportes'
    report_dir.mkdir(parents=True, exist_ok=True)
    output_name = 'T2_EDA_Concreto_Grupo03'

    cmd = [
        'jupyter', 'nbconvert',
        '--to', 'pdf',
        str(out_nb),
        '--output', output_name,
        '--output-dir', str(report_dir)
    ]

    print('Ejecutando nbconvert para generar PDF...')
    try:
        subprocess.run(cmd, check=True)
        pdf_path = report_dir / (output_name + '.pdf')
        if pdf_path.exists():
            print('PDF generado en:', pdf_path)
        else:
            print('nbconvert terminó sin errores, pero no encontré el PDF en:', report_dir)
    except subprocess.CalledProcessError as e:
        print('Error ejecutando nbconvert. Asegúrate de tener Jupyter y una distribución LaTeX instalada.')
        print('Comando:', ' '.join(cmd))
        sys.exit(e.returncode)


if __name__ == '__main__':
    main()
