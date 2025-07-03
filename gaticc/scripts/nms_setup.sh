#!/bin/bash

wget http://galactos.local:8471/nms_dataset.tar.gz

tar -xzvf nms_dataset.tar.gz -C dataset/

wget http://galactos.local:8471/ssd_vgg_35_300_uint8.onnx
wget http://galactos.local:8471/nms_operator.onnx

