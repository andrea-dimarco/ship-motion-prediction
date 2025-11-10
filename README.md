# Adaptive CPS Controller

This repository contains the body of work produced by the collaboration between the departments of **Computer Science** and **Ingegneria Aeronautica e Spaziale** in **Sapienza**


## How to use 

First of all, build the docker image with the following command

```
./build.sh [-h]
```

The you can run any script inside the `/app` folder with the command

```
./run [-r] [script.py]
```

The above command will create a container and run `python /app/[script.py]`. Once the script is done, the container will be removed.  Containers can only access (read and write) the folder `/data`.

If you wish to free up your memory and delete both the images and the lingering containers at once, run the following command

```
./clean.sh
```

To interact with the Cluster use the script

```
./cluster.sh [-h]
```
