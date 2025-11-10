#!/bin/bash

image_name="ship_pred"


echo "Stoping and removing the application containers ..."
docker stop ${image_name}_container
docker rm ${image_name}_container


echo "Removing the appplication docker images ..."
docker rmi ${image_name}:latest

echo "Done."