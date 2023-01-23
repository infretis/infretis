infretisrun -i infretis.toml
infretisanalyze -i infretis_data.txt -p pattern.txt
# pyretisanalyse -i retis.rst # to produce report

# this deletes everything
rm -r 0* out.rst* pyretis.log* pyretis.restart*
rm pattern.txt infretis_data.txt
