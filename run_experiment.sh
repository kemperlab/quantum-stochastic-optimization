#!/bin/bash

r=$2
folder=$1

type=multiple
python -m qso -r $r -n 500 -i -o $folder/$type/$r.json $folder/$type.json

type=expected
python -m qso -r $r -n 500 -i -o $folder/$type/$r.json $folder/$type.json

type=default
python -m qso -r $r -n 500 -i -o $folder/$type/$r.json $folder/$type.json
