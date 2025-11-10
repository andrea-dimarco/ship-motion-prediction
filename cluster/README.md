# Complex RL & Controllers

## Cluster Folder

This folder contains all the scripts to execute experiments on the **Sapienza cluster**

## How to use

To interact with the cluster use the script

```
./cluster.sh [-h]
```

### Run experiments

If you are not using Sapienza's network, you must connect to the VPN

```
./cluster.sh -c
```

Upload the project directory to the cluster with 

```
./cluster.sh -u
```

The connect via ssh to the cluster with

```
./cluster.sh -x
```

Go to the **project folder** and enter the `/cluster` folder, finally compile the **Singularity image** with the following command. Note that this may take some time depending on the size of the *Docker* image.

```
cd {...}/cluster
./cluster.sh -b
```

Now you can run the experiment (the `run.slurm` script) via the command

```
./cluster.sh -r
```

From **your machine**, retrieve the results with
```
./cluster.sh -g
```

You can disconnect from Sapienza's VPN with

```
./cluster.sh -d
```