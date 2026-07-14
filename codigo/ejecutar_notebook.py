from pathlib import Path
import os
import subprocess
import sys

import nbformat
from nbclient import NotebookClient

project_dir = Path(__file__).resolve().parent.parent
results_dir = project_dir / 'resultados'
images_dir = results_dir / 'imagenes'
reports_dir = results_dir / 'reportes'
images_dir.mkdir(parents=True, exist_ok=True)
reports_dir.mkdir(parents=True, exist_ok=True)

os.environ['MPLBACKEND'] = 'Agg'

notebook_path = project_dir / 'T2_EDA_Concreto_Grupo03.ipynb'
out_notebook = results_dir / 'T2_EDA_Concreto_Grupo03_ejecutado.ipynb'
out_html = reports_dir / 'T2_EDA_Concreto_Grupo03.html'
out_pdf = reports_dir / 'T2_EDA_Concreto_Grupo03.pdf'

print('Ejecutando notebook:', notebook_path)
nb = nbformat.read(notebook_path, as_version=4)
client = NotebookClient(nb, timeout=3600, kernel_name='python3', resources={'metadata': {'path': str(project_dir)}})
client.execute()
nbformat.write(nb, out_notebook)
print('Notebook ejecutado guardado en:', out_notebook)

cmd = [
    sys.executable,
    '-m', 'jupyter', 'nbconvert',
    '--to', 'pdf',
    '--output-dir', str(reports_dir),
    '--output', out_pdf.stem,
    str(out_notebook),
]
result = subprocess.run(cmd, cwd=project_dir, capture_output=True, text=True)
print('STDOUT:', result.stdout)
print('STDERR:', result.stderr)
if result.returncode != 0:
    raise SystemExit(result.returncode)
print('HTML guardado en:', out_html)
