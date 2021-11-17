#!/bin/bash

#simple script for resizing images in all class directories
#also reformats everything from whatever to png

cd /Users/Nidhi/Downloads/Blind\ Project/Binary

if [ `ls data/*/*/*.JPG 2> /dev/null | wc -l ` -gt 0 ]; then
for file in data/*/*/*.JPG; do
    convert "$file" -resize 64x64\! "${file%.*}.png"
    file "$file" #uncomment for testing
    #rm "$file"
done
fi


if [ `ls data/*/*/*.jpg 2> /dev/null | wc -l ` -gt 0 ]; then
  for file in data/*/*/*.jpg; do
    convert "$file" -resize 64x64\! "${file%.*}.png"
    file "$file" #uncomment for testing
    #rm "$file"
  done
fi



if [ `ls data/*/*/*.jpeg 2> /dev/null | wc -l ` -gt 0 ]; then
  for file in data/*/*/*.jpeg; do
    convert "$file" -resize 64x64\! "${file%.*}.png"
    file "$file" #uncomment for testing
    #rm "$file"
  done
fi



if [ `ls data/*/*/*.png 2> /dev/null | wc -l ` -gt 0 ]; then
  for file in data/*/*/*.png; do
    convert "$file" -resize 64x64\! "${file%.*}.png"
    file "$file" #uncomment for testing
    #rm "$file"
  done
fi



# Original Code -- look if previous code got messed up
#if [ `ls classes/*/*.png 2> /dev/null | wc -l ` -gt 0 ]; then
#  for file in classes/*/*.png; do
#    convert "$file" -resize 32x32\! "${file%.*}.png"
#    file "$file" #uncomment for testing
#  done
#fi
