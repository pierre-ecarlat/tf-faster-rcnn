# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np

def parse_rec(filename):
  """ Parse a PASCAL VOC xml file """
  tree = ET.parse(filename)
  objects = []
  for obj in tree.findall('object'):
    obj_struct = {}
    obj_struct['name'] = obj.find('name').text
    obj_struct['pose'] = obj.find('pose').text
    obj_struct['truncated'] = int(obj.find('truncated').text)
    obj_struct['difficult'] = int(obj.find('difficult').text)
    bbox = obj.find('bndbox')
    obj_struct['bbox'] = [int(bbox.find('xmin').text),
                          int(bbox.find('ymin').text),
                          int(bbox.find('xmax').text),
                          int(bbox.find('ymax').text)]
    objects.append(obj_struct)

  return objects


def voc_ap(rec, prec, confidence, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  prec = (prec**confidence)
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap


def recoverOrReadAnnotations(cachefile, imagenames):
  if not os.path.isfile(cachefile):
    # load annots
    recs = {}
    for i, imagename in enumerate(imagenames):
      recs[imagename] = parse_rec(annopath.format(imagename))
      if i % 100 == 0:
        print('Reading annotation for {:d}/{:d}'.format(
                                        i + 1, len(imagenames)))
    # save
    print('Saving cached annotations to {:s}'.format(cachefile))
    with open(cachefile, 'w') as f:
      pickle.dump(recs, f)
  else:
    # load
    with open(cachefile, 'r') as f:
      try:
        recs = pickle.load(f)
      except:
        recs = pickle.load(f, encoding='bytes')

  return recs

def extractClassFromRecs(classname, recs, imagenames):
  class_recs = {}
  for imagename in imagenames:
    R = [obj for obj in recs[imagename] if obj['name'] == classname]
    bbox = np.array([x['bbox'] for x in R])
    difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
    det = [False] * len(R)
    class_recs[imagename] = { 'bbox': bbox, 'difficult': difficult, 'det': det }

  return class_recs

def getRelativesForClass(classname):
  vehicles = ['aeroplane', 'bicycle', 'boat', 'bus', 'car' , 
              'motorbike', 'train']
  indoors  = ['bottle', 'chair', 'diningtable', 'pottedplant', 'sofa' , 
              'tvmonitor']
  animals  = ['bird', 'cat', 'cow', 'dog', 'sheep']
  persons  = ['person']
  if   classname in vehicles: r = vehicles # Vehicles
  elif classname in indoors:  r = indoors  # Indoors
  elif classname in animals:  r = animals  # Animals
  elif classname in persons:  r = persons  # Persons
  else:                       r = []
  if len(r) > 0: r.remove(classname)
  return r

def getNumberOfRelevantRecords(class_recs):
  # Number of relevant records, number of positives
  npos = 0
  for rec in class_recs:
    npos += sum(~class_recs[rec]['difficult'])
  return npos


def readDetections(path):
  detections = [line.rstrip('\n').split(' ') for line in open(path)]

  image_ids = [x[0] for x in detections]
  confidence = np.array([float(x[1]) for x in detections])
  BB = np.array([[float(z) for z in x[2:]] for x in detections])

  return image_ids, confidence, BB

def compareBoxes(BB_gt, BB_det):
  # compute overlaps
  # intersection
  ixmin = np.maximum(BB_gt[:, 0], BB_det[0])
  iymin = np.maximum(BB_gt[:, 1], BB_det[1])
  ixmax = np.minimum(BB_gt[:, 2], BB_det[2])
  iymax = np.minimum(BB_gt[:, 3], BB_det[3])
  iw = np.maximum(ixmax - ixmin + 1., 0.)
  ih = np.maximum(iymax - iymin + 1., 0.)
  inters = iw * ih

  # union
  uni = ((BB_det[2] - BB_det[0] + 1.) * (BB_det[3] - BB_det[1] + 1.) +
         (BB_gt[:, 2] - BB_gt[:, 0] + 1.) *
         (BB_gt[:, 3] - BB_gt[:, 1] + 1.) - inters)

  overlaps = inters / uni
  return np.max(overlaps), np.argmax(overlaps)


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False,
             reward_relatives=0.,
             confidence_metric=True):
  """rec, prec, ap = voc_eval(detpath,
                              annopath,
                              imagesetfile,
                              classname,
                              [ovthresh],
                              [use_07_metric])

  Top level function that does the PASCAL VOC evaluation.

  detpath: Path to detections
      detpath.format(classname) should produce the detection results file.
  annopath: Path to annotations
      annopath.format(imagename) should be the xml annotations file.
  imagesetfile: Text file containing the list of images, one image per line.
  classname: Category name (duh)
  cachedir: Directory for caching the annotations
  [ovthresh]: Overlap threshold (default = 0.5)
  [use_07_metric]: Whether to use VOC07's 11 point AP computation
      (default False)
  """
  # assumes detections are in detpath.format(classname)
  # assumes annotations are in annopath.format(imagename)
  # assumes imagesetfile is a text file with each line an image name
  # cachedir caches the annotations in a pickle file

  # first load gt
  if not os.path.isdir(cachedir):
    os.mkdir(cachedir)
  cachefile = os.path.join(cachedir, 'annots.pkl')

  # read list of images
  with open(imagesetfile, 'r') as f:
    lines = f.readlines()
  imagenames = [x.strip() for x in lines]

  # Get all the gt objects
  recs = recoverOrReadAnnotations(cachefile, imagenames)

  # Extract gt objects specific to this class
  class_recs = extractClassFromRecs(classname, recs, imagenames)
  relatives = getRelativesForClass(classname)
  relatives_recs = []
  for rel in relatives:
    relatives_recs.append(extractClassFromRecs(rel, recs, imagenames))

  # Number of relevant records (all the boxes in the gt)
  npos = getNumberOfRelevantRecords(class_recs)
  relatives_npos = []
  for rel_recs in relatives_recs:
    relatives_npos.append(getNumberOfRelevantRecords(rel_recs))

  
  # Read the detection
  detections_path = detpath.format(classname)
  image_ids, confidence, BB = readDetections(detections_path)

  # Go down detections and mark TPs and FPs (as much as detections)
  nd = len(image_ids)
  tp = np.zeros(nd)
  fp = np.zeros(nd)

  # If no detection for this class, returns nothing
  if BB.shape[0] == 0:
    return [], [], 0.0, None

  # Sort everything by confidence
  sorted_ind = np.argsort(-confidence)
  sorted_scores = np.sort(-confidence)
  BB = BB[sorted_ind, :]
  image_ids = [image_ids[x] for x in sorted_ind]

  # Go down dets and mark TPs and FPs
  for d in range(nd):

    # The detection box to compare
    bb = BB[d, :].astype(float)
    
    # For each possible categories (main and relatives)
    for categ in [-1]+range(0, len(relatives_recs)):
      
      # Overlap
      ovmax = -np.inf
      
      # Main class vs relatives
      if categ < 0:
        R = class_recs[image_ids[d]]
      else:
        R = relatives_recs[categ][image_ids[d]]

      # Boxes
      BBGT = R['bbox'].astype(float)

      # Get the max overlap
      if BBGT.size > 0:
        ovmax, jmax = compareBoxes(BBGT, bb)

      # If overlap, this is a true positive
      if ovmax > ovthresh:
        if not R['difficult'][jmax]:
          if not R['det'][jmax]:
            # If main categ detected, +1
            if categ < 0:
              tp[d] = 1.
              R['det'][jmax] = 1

            # If relative detected, +0.3
            else:
              tp[d] = reward_relatives
              fp[d] = 1 - reward_relatives
              R['det'][jmax] = 1
          # If already detected (should not happen)
          else:
            fp[d] = 1.

      # If no overlap, go to the next category


    # If nothing has been found
    if tp[d] == 0 and fp[d] == 0:
      fp[d] = 1.

  # compute precision recall
  fp = np.cumsum(fp)
  tp = np.cumsum(tp)
  rec = tp / float(npos)
  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  if confidence_metric:
    ap = voc_ap(rec, prec, -sorted_scores)
  else:
    ap = voc_ap(rec, prec, np.asarray([1.]*len(sorted_scores)))

  debug_details = { 
    'number_detections': nd, 
    'images_indexes': image_ids, 
    'boxes': BB, 
    'confidences': np.sort(-confidence), 
    'true_positives': tp, 
    'false_positives': fp, 
    'recalls': rec, 
    'precisions': prec
  }

  return rec, prec, ap, debug_details
