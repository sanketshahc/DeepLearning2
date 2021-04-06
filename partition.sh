#!/bin/bash
# move_csv.sh - move XML files based on fields in a CSV 

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

    mv "`echo $FNAME3`" "`echo $destination`"
    # eg mv "BuyingTheView_SunsetCondoToronto_192543_SLING.xml" "3_5_17"
done < $csvfile

