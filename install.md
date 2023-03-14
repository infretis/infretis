python3 -m venv iretisvenv
git clone https://gitlab.com/danielzh/pyretis.git
cd pyretis ; git checkout dz24/dask
python -m pip install -r requirements-dev.txt
python -m pip install -e .
cd ..
git clone git@github.com:dz24/infretis.git
cd infretis ; git checkout dz24/external
pip install pip --upgrade
pip install dask distributed tomli_w
python -m pip install -e .
cd ..
