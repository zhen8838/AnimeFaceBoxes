# AnimeFaceBoxes

Super fast anime face detection (> 200 fps for 512x512 resolution, RTX2080 + i7 9700k)

## Dependencies
- Python 3.6+ (Anaconda)
- PyTorch-1.0 +
- OpenCV3 (Python)

## Usage
- build nms: sh make.sh
- Manual data labeling: LabelFaceBox.py (you can skip this if you have danbooru2018 dataset)
- labeled data: faceboxes (600+ labels for danbooru2018 dataset)
- train: MyTrain.py
- eval: eval.py

![alt text](https://github.com/WynMew/AnimeFaceBoxes/blob/master/out.png)


codes borrowed a lot from [FaceBoxes.PyTorch](https://github.com/zisianw/FaceBoxes.PyTorch)

-----

## export face annotation from movie

```sh
python ./auto_anno_movie.py --output_dir /media/zqh/Documents/jojo-face-landmark \
    --input_dir /media/zqh/Documents/JOJO4 \
    --id 1 \
    --landmark
```

**NOTE** use `--landmark` can export bbox with landmark, then we can use `labelme` adjust annotation.


## use `label_face_landmark.py` or `labelme` rewrite annotation

```sh
python ./label_face_landmark.py --dataset /media/zqh/Documents/jojo-face-landmark/02
```

