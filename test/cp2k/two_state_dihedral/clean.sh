
read -r -p "Are you sure you want to clean this directory? [y/N] " response
case "$response" in
    [yY][eE][sS]|[yY]) 
	rm -r worker* sim.log trajs/ load/ pattern.txt restart.toml infretis* ~/tracer.txt
	git restore load
        ;;
    *)
      	echo not doing anything
        ;;
esac
