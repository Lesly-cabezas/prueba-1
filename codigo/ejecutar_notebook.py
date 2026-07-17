from pathlib import Path
import os
import subprocess
import sys

import nbformat
from nbclient import NotebookClient
from playwright.sync_api import sync_playwright

project_dir = Path(__file__).resolve().parent.parent
results_dir = project_dir / 'resultados'
images_dir = results_dir / 'imagenes'
reports_dir = results_dir / 'reportes'
images_dir.mkdir(parents=True, exist_ok=True)
reports_dir.mkdir(parents=True, exist_ok=True)

# Usar el backend por defecto de Jupyter para que los gráficos se capten en las salidas del notebook.
# os.environ['MPLBACKEND'] = 'Agg'

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

cmd_html = [
    sys.executable,
    '-m', 'jupyter', 'nbconvert',
    '--to', 'html',
    '--output-dir', str(reports_dir),
    '--output', out_html.name,
    str(out_notebook),
]
result_html = subprocess.run(cmd_html, cwd=project_dir, capture_output=True, text=True)
print('HTML export STDOUT:', result_html.stdout)
print('HTML export STDERR:', result_html.stderr)
if result_html.returncode != 0:
    raise SystemExit(result_html.returncode)
print('HTML guardado en:', out_html)

try:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(out_html.resolve().as_uri(), wait_until='networkidle')
        page.pdf(path=str(out_pdf), format='A4', print_background=True)
        browser.close()
    print('PDF guardado en:', out_pdf)
except Exception as exc:
    print(f'No se pudo generar PDF desde HTML: {exc}')
