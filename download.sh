# Download COCO Features

wget -P coco https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip
unzip coco/trainval_36.zip -d coco/
rm coco/trainval_36.zip

# Download VQA Question Answers data
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Train_mscoco.zip -P yolo/
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Val_mscoco.zip -P yolo/

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Train_mscoco.zip -P yolo/
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Val_mscoco.zip -P yolo/

# Unzip and remove zip files
cd yolo
unzip Annotations_Val_mscoco.zip
unzip Annotations_Train_mscoco.zip
unzip Questions_Train_mscoco.zip
unzip Questions_Val_mscoco.zip

rm Annotations_Train_mscoco.zip
rm Annotations_Val_mscoco.zip
rm Questions_Train_mscoco.zip
rm Questions_Val_mscoco.zip
