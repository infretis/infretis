bash clean.sh
rm -r trajs ;
cp -r ../salt_data/trajs/ .
cd trajs ;
cp -r 0 e0
cp -r 1 e1
cp -r 2 e2
cp -r 3 e3
cp -r 4 e4
cp -r 5 e5
cp -r 6 e6
cd .. ;
infretisbm -i infretis.toml >| out.txt
