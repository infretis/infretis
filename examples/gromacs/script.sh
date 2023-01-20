# incredibly stupid way of running this example:
# run this script in terminal by writing "bash script.sh"

cp ../../infretis/help_func.py    .
cp ../../infretis/conv_inf_py.py  .
cp ../../infretis/easy_swap.py    .
cp ../../infretis/infretis.py     .
cp ../../infretis/pattern.py      .
cp ../../infretis/scheduler.py    .

python3 scheduler.py
python3 conv_inf_py.py # to conv infretis_data.txt to 00*/pathensemble.txt
                       # even if we don't do pyretisanalyze atm
python3 pattern.py     # to create pattern

rm -r 0* out.rst* pyretis.log* pyretis.restart*
rm conv_inf_py.py easy_swap.py  infretis.py   
rm pattern.py scheduler.py pattern.txt
rm help_func.py infretis_data.txt ./gromacs_input/topol.tpr
