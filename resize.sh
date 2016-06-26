for name in ./ImageCLEF2013TrainingSet/*/*.jpg; do
    convert -resize 256x256\! $name $name
    echo $name;
done
for name in ./ImageCLEFTestSetGROUNDTRUTH/*/*.jpg; do
    convert -resize 256x256\! $name $name
    echo $name;
done
