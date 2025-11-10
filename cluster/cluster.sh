#!/bin/bash

project_dir="acpsc"
uniroma_user="andrea.dimarco"
cluster_user="adimarco"
research_team="raise"

dockerfile_path="/data1/${cluster_user}/${project_dir}/install/Dockerfile"
version="1"
registry="di.registry:443" # DO NOT touch this unless you know what you are doing


b=false
g=false
r=false
u=false
c=false
d=false
x=false
while getopts ":hbcdgrux" flag; do
 case $flag in
    b) b=true ;;
    c) c=true ;;
    d) d=true ;;
    g) g=true ;;
    r) r=true ;;
    u) u=true ;;
    x) x=true ;;
    h) echo "Arguments:
    [-b] Build Singularity image;
    [-c] Connect to VPN;
    [-d] Disconnect from VPN
    [-g] Get data from cluster;
    [-r] Run slurm script;
    [-u] Upload current directory to cluster;
    [-x] SSH connection to the cluster;"
        exit 0 ;;
    *) echo "Use [-h] for help"
       exit 1 ;;
  esac
done

if [ $u = true ]; then
    sudo echo "Please provide SUDO credentials:"
fi

if [ $c = true ]; then
    echo "Connecting to VPN ... "
    snx -u ${uniroma_user}@uniroma1.it -s castore.uniroma1.it
fi

if [ $u = true ]; then
    echo "UPLOADING project to the cluster"
    cd ../..
    scp -r ${project_dir} cluster.submit.vpn:/data1/${cluster_user}/
    cd ${project_dir}/custer
    echo "Project sent to the cluster"
fi


if [ $b = true ]; then
    # Build docker image on cluster
    cd ..
    docker build -f ${dockerfile_path} -t ${registry}/${research_team}/${cluster_user}/${project_dir}-v${version}:latest .
    cd ./cluster/

    # Upload image to cluster registry
    docker push ${registry}/${research_team}/${cluster_user}/${project_dir}-v${version}:latest

    # Build Singularity image
    singularity build image.sif docker://${registry}/${research_team}/${cluster_user}/${project_dir}-v${version}:latest
fi


if [ $r = true ]; then
    echo "Sending slurm job to cluster"
    sbatch run.slurm
fi



if [ $g = true ]; then
    echo "DOWNLOADING data from the cluster"
    scp -r cluster.submit.vpn:/data1/${cluster_user}/${project_dir}/data ./results
fi

if [ $x = true ]; then
    echo "SSH connection to the cluster"
    ssh cluster.submit.vpn
fi

if [ $d = true ]; then
    echo "Disconnecting from VPN ... "
    snx -d
fi

echo "Script 'cluster.sh' ended gracefully."