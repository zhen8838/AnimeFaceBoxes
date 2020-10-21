import torch
import torch.backends.cudnn as cudnn
import numpy as np
from layers.functions.prior_box import PriorBox
from models.faceboxes import FaceBoxes
from utils.box_utils import decode
import cv2
from eval import load_model
from data.config import cfg
from utils.nms_wrapper import nms
from typing import TypedDict, List, Dict, AnyStr, Any, Tuple

LANDMARKS = ['left_eye', 'right_eye',
             'nose', 'left_mouth',
             'right_mouth']


class FaceBoxesPredict(object):
  def __init__(self, weightfile: str,
               confidenceTh=0.05,
               nmsTh=0.3,
               keepTopK=750,
               top_k=5000,
               cpu=False) -> None:
    self.confidenceTh = confidenceTh
    self.nmsTh = nmsTh
    self.keepTopK = keepTopK
    self.top_k = top_k
    self.cpu = cpu
    self.net = load_model(FaceBoxes(phase='test', size=None, num_classes=2), weightfile, cpu)
    self.net.eval()
    torch.backends.cudnn.benchmark = True
    self.device = torch.device("cpu" if cpu else "cuda")
    self.net = self.net.to(self.device)

  def predict_one(self, imgOrig: np.ndarray):
    """ predict one image

    Args:
        imgOrig (np.ndarray): cv2 image, NOTE color need be bgr

    Returns:
        np.ndarray: dets
    """
    img = np.float32(imgOrig)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([im_width, im_height, im_width, im_height])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(self.device)
    scale = scale.to(self.device)

    loc, conf = self.net(img)  # forward pass
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(self.device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.data.cpu().numpy()[:, 1]

    # ignore low scores
    inds = np.where(scores > self.confidenceTh)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:self.top_k]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    #keep = py_cpu_nms(dets, args.nms_threshold)
    keep = nms(dets, self.nmsTh, force_cpu=self.cpu)
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:self.keepTopK, :]
    # NOTE dets  n*[xmin,ymin,xmax,ymax,score]
    # for k in range(dets.shape[0]):
    #   xmin = dets[k, 0]
    #   ymin = dets[k, 1]
    #   xmax = dets[k, 2]
    #   ymax = dets[k, 3]
    #   ymin += 0.2 * (ymax - ymin + 1)
    #   score = dets[k, 4]
    # print('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
    #     image_path, score, xmin, ymin, xmax, ymax))
    # cv2.rectangle(imgOrig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 15)

    # pass
    return dets


class AnnoationSub(object):
  label: str
  points: List[Tuple[float, float]]
  group_id: int
  shape_type: str
  flags: Dict


class AnnoationBase(object):
  version: str
  flags: Dict
  shapes: List[AnnoationSub]
  imagePath: str
  imageData: str
  imageHeight: str
  imageWidth: str


def get_face_annotation(xmin: float, ymin: float,
                        xmax: float, ymax: float,
                        group_id: int) -> AnnoationSub:
  return {'label': 'face',
          'points': [[xmin.item(), ymin.item()],
                     [xmax.item(), ymax.item()]],
          'group_id': group_id,
          'shape_type': 'rectangle',
          'flags': {}}


def get_point_annotation(x: float, y: float,
                         label: str,
                         group_id: int) -> AnnoationSub:
  return {'label': label,
          'points': [[x, y]],
          'group_id': group_id,
          'shape_type': 'point',
          'flags': {}}


def get_base_annotation(path: str, Height: int, Width: int) -> AnnoationBase:
  return {'version': '4.5.6',
          'flags': {},
          'shapes': [],
          'imagePath': path,
          'imageData': None,
          'imageHeight': Height,
          'imageWidth': Width}


if __name__ == "__main__":
  predictor = FaceBoxesPredict('FaceBoxes_epoch_90.pth', confidenceTh=0.7)
  im = cv2.imread('/home/zqh/Pictures/jojo-3.jpg', cv2.IMREAD_COLOR)

  dets = predictor.predict_one(im)

  import json
  # with open('/home/zqh/Pictures/jojo-2.json', 'r') as f:
  #   anno: dict = json.loads(f.read())
  anno = get_base_annotation('jojo-3.jpg', *im.shape[:2])
  for group_id, det in enumerate(dets):
    face = get_face_annotation(*det[:-1], group_id)
    anno['shapes'].append(face)

  text = json.dumps(anno, indent=4)
  with open('', 'w') as f:
    f.write(text)
