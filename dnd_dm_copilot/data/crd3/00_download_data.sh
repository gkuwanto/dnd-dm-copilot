cd data
if [ ! -f aligned\ data.zip ]
then wget https://huggingface.co/datasets/crd3/resolve/72bffe55b4d5bf19b530d3e417447b3384ba3673/data/aligned%20data.zip
fi
unzip aligned\ data.zip
