#!/bin/bash

script_dir=`dirname "$0"`
image_name="acpsc"
sound_duration=0.125 # milliseconds
sound_frequency=196 # Hz (C 3rd Octave)



echo "Running containter I for ${image_name} ... "
docker run -it --volume "${script_dir}/app":"/app" \
               --volume "${script_dir}/data":"/data" \
               --name ${image_name}_container ${image_name}:latest \
               python3 /app/"${@}"

echo "Execution ended."


docker rm ${image_name}_container &> /dev/null
echo "Container I for ${image_name} removed."

play -nq -t alsa synth ${sound_duration} sine ${sound_frequency}
