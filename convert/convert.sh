#!/bin/sh
python ./dump_hdf5.py --modality=rgb_imagenet
python ./dump_hdf5.py --modality=rgb_scratch
python ./dump_hdf5.py --modality=flow_imagenet
python ./dump_hdf5.py --modality=flow_scratch

python ./convert_pytorch.py --modality=rgb_imagenet
python ./convert_pytorch.py --modality=rgb_scratch
python ./convert_pytorch.py --modality=flow_imagenet
python ./convert_pytorch.py --modality=flow_scratch

rm ../data/dump -r