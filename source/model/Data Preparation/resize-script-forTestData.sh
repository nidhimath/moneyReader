#!/bin/bash

#simple script for resizing images in all class directories
#also reformats everything from whatever to png

cd /Users/Nidhi/Downloads/Blind Project/Binary

if [ `ls data/*/*/*.JPG 2> /dev/null | wc -l ` -gt 0 ]; then
  for file in test_data/*/*/*.jpg; do
    convert "$file" -resize 256x256\! "${file%.*}.png"
    file "$file" #uncomment for testing
    #rm "$file"
  done
fi


if [ `ls data/*/*/*.jpg 2> /dev/null | wc -l ` -gt 0 ]; then
  for file in test_data/*/*/*.jpg; do
    convert "$file" -resize 256x256\! "${file%.*}.png"
    file "$file" #uncomment for testing
    #rm "$file"
  done
fi




if [ `ls test_data/*/*.png 2> /dev/null | wc -l ` -gt 0 ]; then
  for file in test_data/*/*.png; do
    convert "$file" -resize 256x256\! "${file%.*}.png"
    file "$file" #uncomment for testing
  done
fi
