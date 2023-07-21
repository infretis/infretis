python3 -m venv iretisvenv
source iretisvenv/bin/activate
git clone https://github.com/infretis/infretis.git
cd infretis
pip install pip --upgrade
pip install dask==2023.3.0 distributed==2023.3.0 tomli numpy tomli_w matplotlib
python -m pip install -e .
