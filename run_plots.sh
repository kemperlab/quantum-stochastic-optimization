#!/bin/bash

folder=$1
mkdir -p plots/$folder/
python -m qso.plot -f data/$folder/{multiple,expected,default}  --names multiple expected default -p data/$folder/default.json -n 1 -o plots/$folder\_costs.pdf
