#!/bin/bash

script_dir=`dirname "$0"`
image_name="acpsc"

v=false
while getopts ":hv" flag; do
 case $flag in
    v) v=true ;;
    h) echo "Arguments:
    [-v] Add /app and /data volumes to the image"
    exit 0 ;;
    *) echo "Use [-h] for help"
       exit 1 ;;
  esac
done

echo "Building the image for ${image_name} ... "
docker build . -t ${image_name}:latest -f install/Dockerfile
docker images ${image_name}


if [ $v = true ]; then
    echo "Adding volumes to image ..."
    docker container create --volume "${script_dir}/app":"/app" \
                            --volume "${script_dir}/data":"/data" \
                            --name ${image_name}_container rlc:latest
    docker commit ${image_name}_container ${image_name}:latest
    docker rm ${image_name}_container &> /dev/null
fi

echo "Image has been built and is ready to be used."