from eval_base import FaceBoxesPredict, get_base_annotation, get_face_annotation, get_point_annotation, LANDMARKS
import json
import cv2
import os
import re
import glob
import argparse
import numpy as np

TEMPLATE = np.array([[0.34191607, 0.46157411], [0.65653393, 0.45983393],
                     [0.500225, 0.64050536],
                     [0.37097589, 0.82469196], [0.631517, 0.82325089]])


def main(output_dir, input_dir, idx, landmark: bool):
  # output_dir = '/media/zqh/Documents/jojo-face-landmark'
  # input_dir = '/media/zqh/Documents/JOJO4'
  movie_paths = glob.glob(input_dir + '/*.mp4')
  patten = re.compile(
      '\[AGE-JOJO&UHA-WING&Kamigami\]\[160073\]\[(\d+)\]\[BD-720P\]\[CHS-JAP\] AVC.mp4')
  predictor = FaceBoxesPredict('FaceBoxes_epoch_90.pth', confidenceTh=0.7)
  movie_path = movie_paths[idx]
  movie_id = patten.findall(movie_path)[0]
  stream = cv2.VideoCapture(movie_path)
  n = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
  output_path = os.path.join(output_dir, movie_id)
  if not os.path.exists(output_path):
    os.mkdir(output_path)
  print(movie_path, 'Frames = ', n)
  while True:
    ret, frame = stream.read()
    frame_id = int(stream.get(cv2.CAP_PROP_POS_FRAMES))
    if not ret:
      break
    """ do some thing """
    dets = predictor.predict_one(frame)
    if len(dets) > 0:
      frame_name = f'{frame_id:d}'
      image_name = frame_name + '.jpg'
      json_name = frame_name + '.json'
      anno = get_base_annotation(image_name, *frame.shape[:2])
      for group_id, det in enumerate(dets):
        face = get_face_annotation(*det[:-1], group_id)
        anno['shapes'].append(face)
        if landmark:  # NOTE 导出landmark
          xymin = np.array(face['points'][0])
          xymax = np.array(face['points'][1])
          wh = xymax - xymin
          for scale, label in zip(TEMPLATE, LANDMARKS):
            x, y = xymin + (wh * scale)
            point = get_point_annotation(x.item(), y.item(), label, group_id)
            anno['shapes'].append(point)

      text = json.dumps(anno, indent=4)
      with open(os.path.join(output_path, json_name), 'w') as f:
        f.write(text)
      cv2.imwrite(os.path.join(output_path, image_name), frame)
      """ skip frame """
    stream.set(cv2.CAP_PROP_POS_FRAMES, frame_id + 30)
  stream.release()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='')

  parser.add_argument('--output_dir', default='/media/zqh/Documents/jojo-face-landmark', type=str)
  parser.add_argument('--input_dir', default='/media/zqh/Documents/JOJO4', type=str)
  parser.add_argument('--id', default=1, type=int)
  parser.add_argument('--landmark', action='store_true')

  args = parser.parse_args()
  main(args.output_dir, args.input_dir, args.id, args.landmark)
