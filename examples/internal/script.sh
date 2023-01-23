# incredibly stupid way of running this example:
# run this script in terminal by writing "bash script.sh"

# cp ../../infretis/conv_inf_py.py  .
# cp ../../infretis/infretis.py     .
# cp ../../infretis/pattern.py      .
# cp ../../infretis/scheduler.py    .

infretisrun -i infretis.toml
# python3 conv_inf_py.py # to conv infretis_data.txt to 00*/pathensemble.txt
#                        # even if we don't do pyretisanalyze atm
# python3 pattern.py     # to create pattern

rm -r 0* out.rst* pyretis.log* pyretis.restart*
rm pattern.txt infretis_data.txt
