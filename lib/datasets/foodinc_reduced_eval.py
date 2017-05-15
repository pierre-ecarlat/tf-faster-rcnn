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

#    print(objects)

    return objects

def foodinc_reduced_ap(rec, prec):
    """ ap = foodinc_reduced_ap(rec, prec)
    Compute Foodinc AP given precision and recall.
    """
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

def foodinc_reduced_eval(detpath,
                         annopath,
                         imagesetfile,
                         classname,
                         class_id,
                         cachedir,
                         ovthresh=0.5):
    """rec, prec, ap = foodinc_eval(detpath,
                                    annopath,
                                    imagesetfile,
                                    classname,
                                    class_id,
                                    [ovthresh])

    Top level function that does the Foodinc reduced evaluation.

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
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            try:
                recs = pickle.load(f)
            except:
                recs = pickle.load(f, encoding='bytes')

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == str(class_id)]
        bbox = np.array([x['bbox'] for x in R])
        # Keep track of counted detections, initialise as false
        det = [False] * len(R)
        # Number of relevant records, number of positives
        npos = npos + len(R)
        class_recs[imagename] = {'bbox': bbox,
                                 'det': det}

    # read dets
    detfile = detpath.format(class_id)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    if BB.shape[0] > 0:
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        for d in range(nd):
          R = class_recs[image_ids[d]]
          bb = BB[d, :].astype(float)
          ovmax = -np.inf
          BBGT = R['bbox'].astype(float)

          if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    ap = foodinc_reduced_ap(rec, prec)

    return rec, prec, ap