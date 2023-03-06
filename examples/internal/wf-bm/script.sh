bash clean.sh
cp ../../../test/examples/internal/data/wf.rst retis.rst
cp ../../../test/examples/internal/data/initial.xyz initial.xyz
rm -r trajs ;
mkdir trajs ;
cp -r ../../../test/examples/internal/data/trajs/ .
cd trajs ;
cp -r 0 e0
cp -r 1 e1
cp -r 2 e2
cp -r 3 e3
cp -r 4 e4
cp -r 5 e5
cp -r 6 e6
cp -r 7 e7
cd ..
infretisbm -i infretis.toml >| out.txt
