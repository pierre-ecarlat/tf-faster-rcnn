# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import re
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import pickle
import subprocess
import time
import uuid
from .foodinc_eval import foodinc_eval
from model.config import cfg


class foodinc(imdb):
  def __init__(self, image_set, year, devkit_path=None):
    imdb.__init__(self, 'foodinc_' + year + '_' + image_set)
    self._year = year
    self._image_set = image_set
    self._devkit_path = self._get_default_path() if devkit_path is None \
      else devkit_path
    self._data_path = os.path.join(self._devkit_path)
    self._classes = ('__background__',  # always index 0
                     'white rice', 'brown rice', 'other rice', 
                     'bread', 'cooked bread', 'sweet bread', 'sandwich', 'other bread', 
                     'ramen', 'udon', 'soba', 'other noodles', 
                     'white fish', 'blue fish', 'shellfish', 'crutacean', 'other fish', 
                     'steak', 'beef', 'pork', 'chicken', 'ham - baccon', 'other meat', 
                     'tofu', 'soymilk', 'natto', 'beans', 'other soyfoods', 
                     'eggs', 'egg dish', 
                     'fruits', 
                     'tomato', 'broccoli', 'root crops', 'green and yellow vegetables', 'mushrooms', 'other vegetables', 
                     'milk', 'yogurt', 'cheese', 'other dairy products', 
                     'nuts & seeds', 
                     'water', 'juice', 'vegetable juice', 'coffe - tea', 'alcohol', 'other drinks', 
                     'stir-fried food', 'fried food', 'steamed food', 'grilled food', 'simmered food', 
                     'green salad', 'seaweed salad', 'potato - pumpkins salad', 'proteinized salad', 'vinegared salad', 
                     'soup stock', 
                     'pastry', 'japanese pastry', 
                     'curry', 'rice ball - seaweed roll', 'porridge', 'other rice dishes', 
                     'pot', 'bento')
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    self._image_ext = '.png'
    self._image_index = self._load_image_set_index()
    # Default to roidb handler
    self._roidb_handler = self.gt_roidb
    self._salt = str(uuid.uuid4())
    self._comp_id = 'comp4'

    # PASCAL specific config options
    self.config = {'cleanup': True,
                   'use_salt': True,
                   'use_diff': False,
                   'matlab_eval': False,
                   'rpn_file': None,
                   'debug': False}

    assert os.path.exists(self._devkit_path), \
      'Foodinc path does not exist: {}'.format(self._devkit_path)
    assert os.path.exists(self._data_path), \
      'Path does not exist: {}'.format(self._data_path)

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    image_path = os.path.join(self._data_path, 'Images',
                              index + self._image_ext)
    assert os.path.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def _load_image_set_index(self):
    """
    Load the indexes listed in this dataset's image set file.
    """
    # Example path to image set file:
    # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
    image_set_file = os.path.join(self._data_path, 'ImageSets',
                                  self._image_set + '.txt')
    assert os.path.exists(image_set_file), \
      'Path does not exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
      image_index = [x.strip() for x in f.readlines()]
    return image_index

  def _get_default_path(self):
    """
    Return the default path where PASCAL VOC is expected to be installed.
    """
    return os.path.join(cfg.DATA_DIR, 'Foodinc_' + self._year)

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        try:
          roidb = pickle.load(fid)
        except:
          roidb = pickle.load(fid, encoding='bytes')
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    gt_roidb = [self._load_foodinc_annotation(index)
                for index in self.image_index]
    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))

    return gt_roidb

  def rpn_roidb(self):
    gt_roidb = self.gt_roidb()
    rpn_roidb = self._load_rpn_roidb(gt_roidb)
    roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
    
    return roidb

  def _load_rpn_roidb(self, gt_roidb):
    filename = self.config['rpn_file']
    print('loading {}'.format(filename))
    assert os.path.exists(filename), \
      'rpn data not found at: {}'.format(filename)
    with open(filename, 'rb') as f:
      box_list = pickle.load(f)
    return self.create_roidb_from_box_list(box_list, gt_roidb)

  def _load_foodinc_annotation(self, index):
    """
    Load image and bounding boxes info from annotation text file in the format:
    obj1_class_id x1 y1 x2 y2
    obj2_class_id x1 y1 x2 y2
    obj3_class_id x1 y1 x2 y2
    """
    # For XML case, refer to pascalvoc.py
    filename = os.path.join(self._data_path, 'Annotations', index + '.txt')

    with open(filename) as f:
      data = f.read()
    
    # import re
    objs = re.findall('\d+[\s\-]+\d+[\s\-]+\d+[\s\-]+\d+[\s\-]+\d+', data)
    
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    # area of bounding box
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
      coor = re.findall('\d+', obj)
      # Make pixel indexes 0-based
      cls = int(coor[0])
      x1 = float(coor[1])
      y1 = float(coor[2])
      x2 = float(coor[3])
      y2 = float(coor[4])

      # Fix negative coords
      if x1 < 0:
        x1 = 0
      if y1 < 0:
        y1 = 0
      if x1 > x2:
        print('Problem with annotation in {}'.format(filename))

      boxes[ix, :] = [x1, y1, x2, y2]
      gt_classes[ix] = cls
      overlaps[ix, cls] = 1.0
      seg_areas[ix] = (x2 - x1) * (y2 - y1)

    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes' : boxes,
            'gt_classes': gt_classes,
            'gt_overlaps' : overlaps,
            'flipped' : False,
            'seg_areas' : seg_areas}

  def _get_comp_id(self):
    comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
               else self._comp_id)
    return comp_id

  def _get_foodinc_results_file_template(self):
    # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    filename = self._get_comp_id() + '_det_' + self._image_set + '_{:07d}.txt'
    path = os.path.join(
      self._devkit_path,
      'results',
      'Foodinc_' + self._year,
      filename)
    return path

  def _write_foodinc_results_file(self, all_boxes):
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      print('Writing {} Foodinc results file, ID: {}'.format(cls, cls_ind))
      filename = self._get_foodinc_results_file_template().format(cls_ind)
      with open(filename, 'wt') as f:
        for im_ind, index in enumerate(self.image_index):
          dets = all_boxes[cls_ind][im_ind]
          if dets == []:
            continue
          for k in range(dets.shape[0]):
            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                    format(index, dets[k, -1],
                           dets[k, 0], dets[k, 1],
                           dets[k, 2], dets[k, 3]))

  def _write_foodinc_debug_file(self, all_boxes, debug_dir):
    annotations = os.path.join(debug_dir, 'annotations')

    for im_ind, index in enumerate(self.image_index):
      with open(os.path.join(annotations, index + '.txt'), 'a') as f:
        for cls_ind, cls in enumerate(self.classes):
          if cls == '__background__':
            continue
          dets = all_boxes[cls_ind][im_ind]
          if dets == []:
            continue
          for k in range(dets.shape[0]):
            f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
                    format(cls_ind, 
                           dets[k, 0], dets[k, 1], 
                           dets[k, 2], dets[k, 3], 
                           dets[k, -1]))
  
  def _do_python_eval(self, output_dir='output'):
    annopath = os.path.join(
      self._data_path,
      'Annotations',
      '{:s}.txt')
    imagesetfile = os.path.join(
      self._data_path,
      'ImageSets',
      self._image_set + '.txt')
    cachedir = os.path.join(self._devkit_path, 'annotations_cache')
    aps = []
    if not os.path.isdir(output_dir):
      os.mkdir(output_dir)
    # mini_val = [1, 12, 36, 2]
    for i, cls in enumerate(self._classes):
      # if i not in mini_val:
      #   continue
      if cls == '__background__':
        continue
      filename = self._get_foodinc_results_file_template().format(i)
      rec, prec, ap = foodinc_eval(
          filename, annopath, imagesetfile, cls, i, cachedir, ovthresh=0.5)
      aps += [ap]
      print('AP for {} = {:.4f}'.format(cls, ap))
      with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
        pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

    print(('Mean AP = {:.4f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
      print(('{:.3f}'.format(ap)))
    print(('{:.3f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    print('-- Thanks, The Management')
    print('--------------------------------------------------------------')

  def _do_matlab_eval(self, output_dir='output'):
    print('--------------------------------------------------------------')
    print('Not implemented...')
    print('--------------------------------------------------------------')

  def evaluate_detections(self, all_boxes, output_dir):
    if self.config['debug']:
      right_now = time.strftime("%Y%m%d%H%M%S")
      debug_dir = os.path.join(cfg.ROOT_DIR, 'debug', date)
      if not os.path.isdir(debug_dir):
        os.makedirs(debug_dir)

    self._write_foodinc_results_file(all_boxes)
    if self.config['debug']:
      self._write_foodinc_debug_file(all_boxes, debug_dir)
    self._do_python_eval(output_dir)
    if self.config['matlab_eval']:
      self._do_matlab_eval(output_dir)
    if self.config['cleanup']:
      for i, cls in enumerate(self._classes):
        if cls == '__background__':
          continue
        filename = self._get_foodinc_results_file_template().format(i)
        os.remove(filename)

  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True

  def debug(self, on):
    if on:
      self.config['debug'] = True
    else:
      self.config['debug'] = False


if __name__ == '__main__':
  from datasets.foodinc import foodinc

  d = foodinc('trainval', '2017')
  res = d.roidb
  from IPython import embed;

  embed()
