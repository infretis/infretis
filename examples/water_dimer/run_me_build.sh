
rm -rf ??? amsworker_* out.rst* pyretis.log* error.txt* EXIT initial pyretis.restart

source ~/opt/ams/ams2023.101/amsbashrc.sh

(
    cd ~/.scm/python/AMS2023.1.venv/lib/python3.8/site-packages/pyretis/
    amspython -m pip install .
)

amspython ~/.scm/python/AMS2023.1.venv/lib/python3.8/site-packages/pyretis/pyretis/bin/pyretisrun.py -l debug -i pyretis.in

