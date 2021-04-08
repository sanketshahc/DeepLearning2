#!/bin/bash 
# Downloads pets data and organizes it into folders for torch Dataset object.

wget -O- https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz | tar -xvz -C resources/

MYDIR="${PWD}";

filename_field=1;
from_dir=${MYDIR}/resources/images/
destination=${MYDIR}/resources/test_images/;
csvfile=${MYDIR}/resources/annotations/test.txt;

while read csv_line; 
do
    # echo $csv_line
    FNAME=$(echo $csv_line | cut -d ' ' -f $filename_field);
    FNAME2=$(echo $FNAME.jpg)
    FNAME3=$(echo $from_dir$FNAME2)
    mkdir -pv ${destination};
    mv "`echo $FNAME3`" "`echo $destination`";
    # eg mv "BuyingTheView_SunsetCondoToronto_192543_SLING.xml" "3_5_17"
done  < $csvfile

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


for FLE in ${MYDIR}/resources/images/*.jpg;
do
    FNAME=$(echo $(basename ${FLE}) | sed 's/_[0-9].*//g');
    # echo $FLE
    DEST="${MYDIR}/resources/test_images/${FNAME}"
    # printf $FLE
    mkdir -pv ${DEST};
    # echo ${DEST}
    mv ${FLE} ${DEST};

done
