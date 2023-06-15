for i in *;
do
	cd $i/accepted
	for file in *.xyz
	do
		in=$file
		out=$(echo $file | sed 's/xyz/gro/' )
		obabel -ixyz $in -ogro -O $out
		sed "s/$in/$out/g" ../traj.txt -i
		sed 's/0\.00000/0\.80000/g' $out -i
	done
	rm *.xyz
	cd -
done
