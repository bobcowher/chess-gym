#!/bin/bash

apt-get update -y
apt-get install libcairo2-dev -y

pip install -r requirements.txt