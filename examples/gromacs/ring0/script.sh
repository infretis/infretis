bash clean.sh
rm -r trajs ;
cp -r ../../../test/examples/external/data/trajs/ .
cd trajs ;
cp -r 0 e0
cp -r 1 e1
cp -r 2 e2
cd .. ;
infretisrun -i infretis.toml >| out.txt
# infretisanalyze -i infretis_data.txt -p pattern.txt
# # pyretisanalyse -i retis.rst # to produce report
