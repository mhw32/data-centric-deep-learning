#!/bin/bash

curl -X 'POST' \
  'http://127.0.0.1:8000/api/v1/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "image_url": "https://conx.readthedocs.io/en/latest/_images/MNIST_44_0.png"
  }'
