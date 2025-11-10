# Manually Update Image
  
  **Step 1**: Build the image, in this example it will be called `image_name`
  
  **Step 2**: Run a container of the image with interactive bash
  ```
  docker run -i --name update_container image_name bash
  ```
  
  **Step 3**: From within the `update_container` perform all the edits you desire
  
  **Step 4**: Exit the `update_container` bash by typing
  ```
  exit
  ```
  
  **Step 5**: Now turn the container into an image with the following command. Note that if you give the new image the name of an existing image then that *old* image will be **renamed with a hash** and the provided name will be given to the *new* image
  ```
  docker commit update_container new_image_name
  ```

  **Step 6**: Optionally, delete the old image with the command
  ```
  docker image rm image_name
  ```