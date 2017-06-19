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
import re

def parse_rec(filename):
  """ Parse a Foodinc txt file """
  
  print('Reading annotation for file: {}'.format(filename))
  
  with open(filename) as f:
    data = f.read()
  
  # import re
  objs = re.findall('\d+[\s\-]+\d+[\s\-]+\d+[\s\-]+\d+[\s\-]+\d+', data)
  num_objs = len(objs)

  # Return objects
  objects = []

  # Load object bounding boxes into a data frame.
  for ix, obj in enumerate(objs):
    coor = re.findall('\d+', obj)
    # Make pixel indexes 0-based
    cls = int(coor[0])
    x1 = float(coor[1])
    y1 = float(coor[2])
    x2 = float(coor[3])
    y2 = float(coor[4])

    obj_struct = {}
    obj_struct['name'] = str(cls)
    obj_struct['bbox'] = [int(x1),
                          int(y1),
                          int(x2),
                          int(y2)]
    objects.append(obj_struct)

#  print(objects)

  return objects

def foodinc_ap(rec, prec, confidence):
  """ ap = foodinc_ap(rec, prec)
  Compute Foodinc AP given precision and recall.
  """
  # correct AP calculation
  prec = (prec**confidence)
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

def extractClassFromRecs(class_id, recs, imagenames):
  class_recs = {}
  for imagename in imagenames:
    R = [obj for obj in recs[imagename] if obj['name'] == str(class_id)]
    bbox = np.array([x['bbox'] for x in R])
    # Keep track of counted detections, initialise as false
    det = [False] * len(R)
    class_recs[imagename] = { 'bbox': bbox, 'det': det }

  return class_recs

def getRelativesForClass(class_id):
  if   1  <= class_id <= 3:  r = range(1,  3  + 1) # Rice
  elif 4  <= class_id <= 8:  r = range(4,  8  + 1) # Bread
  elif 9  <= class_id <= 12: r = range(9,  12 + 1) # Noodles
  elif 13 <= class_id <= 17: r = range(13, 17 + 1) # Fish
  elif 18 <= class_id <= 23: r = range(18, 23 + 1) # Meat
  elif 24 <= class_id <= 28: r = range(24, 28 + 1) # Soyfood
  elif 29 <= class_id <= 30: r = range(29, 30 + 1) # Eggs
  elif 31 <= class_id <= 31: r = range(31, 31 + 1) # Fruits
  elif 32 <= class_id <= 37: r = range(32, 37 + 1) # Vegetables
  elif 38 <= class_id <= 41: r = range(38, 41 + 1) # Dairy products
  elif 42 <= class_id <= 42: r = range(42, 42 + 1) # Nuts and seeds
  elif 43 <= class_id <= 48: r = range(43, 48 + 1) # Beverages
  elif 49 <= class_id <= 53: r = range(49, 53 + 1) # Recipes
  elif 54 <= class_id <= 58: r = range(54, 58 + 1) # Salads
  elif 59 <= class_id <= 59: r = range(59, 59 + 1) # Soup stocks
  elif 60 <= class_id <= 61: r = range(60, 61 + 1) # Pastry
  elif 62 <= class_id <= 65: r = range(62, 65 + 1) # Rice dishes
  elif 66 <= class_id <= 67: r = range(66, 67 + 1) # Others
  else:                      r = []
  if len(r) > 0: r.remove(class_id)
  return r


def getNumberOfRelevantRecords(class_recs):
  # Number of relevant records, number of positives
  npos = 0
  for rec in class_recs:
    npos += len(class_recs[rec]['det'])
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


def foodinc_eval(detpath,
                 annopath,
                 imagesetfile,
                 classname,
                 class_id,
                 cachedir,
                 ovthresh=0.5,
                 reward_relatives=0.,
                 confidence_metric=False):
  """rec, prec, ap = foodinc_eval(detpath,
                                  annopath,
                                  imagesetfile,
                                  classname,
                                  class_id,
                                  [ovthresh])

  Top level function that does the Foodinc evaluation.

  detpath: Path to detections
      detpath.format(classname) should produce the detection results file.
  annopath: Path to annotations
      annopath.format(imagename) should be the xml annotations file.
  imagesetfile: Text file containing the list of images, one image per line.
  classname: Category name (duh)
  cachedir: Directory for caching the annotations
  [ovthresh]: Overlap threshold (default = 0.5)
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
  class_recs = extractClassFromRecs(class_id, recs, imagenames)
  relatives = getRelativesForClass(class_id)
  relatives_recs = []
  for rel in relatives:
    relatives_recs.append(extractClassFromRecs(rel, recs, imagenames))

  # Number of relevant records (all the boxes in the gt)
  npos = getNumberOfRelevantRecords(class_recs)
  relatives_npos = []
  for rel_recs in relatives_recs:
    relatives_npos.append(getNumberOfRelevantRecords(rel_recs))

  
  # Read the detection
  detections_path = detpath.format(class_id)
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
    ap = foodinc_ap(rec, prec, -sorted_scores)
  else:
    ap = foodinc_ap(rec, prec, np.asarray([1.]*len(sorted_scores)))

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

