#!/bin/bash

MYDIR="${PWD}";

for FLE in ${MYDIR}/resources/test_images/*.jpg;
do
FNAME=$(echo $(basename ${FLE}) | sed 's/_[0-9].*//g');
# echo $FLE
DEST="${MYDIR}/resources/test_images/${FNAME}"
# printf $FLE
mkdir -pv ${DEST};
# echo ${DEST}

mv ${FLE} ${DEST};

done
