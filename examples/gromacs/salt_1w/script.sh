bash clean.sh
rm -r trajs ;
cp -r ../salt_data2/trajs/ .
cd trajs ;
cp -r 0 e0
cp -r 1 e1
cp -r 2 e2
cp -r 3 e3
cp -r 4 e4
cp -r 5 e5
cp -r 6 e6
cd .. ;
infretisrun -i infretis.toml >| out.txt
# infretisanalyze -i infretis_data.txt -p pattern.txt
# # pyretisanalyse -i retis.rst # to produce report
