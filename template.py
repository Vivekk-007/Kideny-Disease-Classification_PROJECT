import os
from pathlib import Path
import logging

# logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

project_name = 'CNN_Classifier'

list_of_files = [
    '.github/workflows/.gitkeep',
    'src/{project_name}/__init__.py',
    'src/{project_name}/components/__init__.py',
    'src/{project_name}/utils/__init__.py',
    'src/{project_name}/config/__init__.py',
    'src/{project_name}/config/configuration.py',
    'src/{project_name}/pipeline/__init__.py',
    'src/{project_name}/entity/__init__.py',
    'src/{project_name}/constants/__init__.py',
    'config/config.yaml',
    'dvc.yaml',
    'params.yaml',
    'app.py',
    'main.py',
    'Dockerfile',
    'requirements.txt',
    'setup.py',
    'research/trials.ipynb'
    'templates/index.html'
]


for file_path in list_of_files:
    file_path = file_path.format(project_name=project_name)
    logging.info(f'Creating file: {file_path}')
    path_obj = Path(file_path)
    dir_name = path_obj.parent

    try:
        if not dir_name.exists():
            dir_name.mkdir(parents=True, exist_ok=True)
            logging.info(f'Created directory: {dir_name}')

        if not path_obj.exists() or path_obj.stat().st_size == 0:
            with open(path_obj, 'w') as f:
                pass
            logging.info(f'Created file: {file_path}')
        else:
            logging.info(f'File already exists and is not empty: {file_path}')
    except Exception as e:
        logging.error(f'Error creating file {file_path}: {e}')

        

