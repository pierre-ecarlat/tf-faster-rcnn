#!/usr/bin/env python

"""
????
CMD
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import xml.etree.ElementTree as ET
import sys
import os
import argparse
import cv2

import Tkinter
from Tkinter import *
import Tkconstants
import Image, ImageTk


# Max sizes for an image
MAX_WIDTH = 600
MAX_HEIGHT = 400

_paths = {}

_nbimages = 0
_imageidx = 0
_allimages = []
_allannotations = []
_alldetections = []
_threshold = 0.

_categs_details = []
_categoryidx = 0
_detectionidx = []



def getArguments():
    """Defines and parses command-line arguments to this script."""
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('dataset', help='\
    The directory with the dataset (should have Images/ and Annotations/).')
    parser.add_argument('results_dir', help='\
    The directory with the results.')
    parser.add_argument('--ann_format', default="txt", help='\
    The format of the annotation (txt and xml supported).')
    parser.add_argument('--img_format', default="png", help='\
    The format of the annotation (txt and xml supported).')
    parser.add_argument('--threshold', default=0., type=float, help='\
    The threshold for vcisualization, between 0 and 1, 0.3 is the default value.')
    parser.add_argument('--all', default=False, action='store_true', help='\
    Average of all the annotations.')

    return parser.parse_args()


def getColors(dataset_path):
    colors = [(0,255,0) for _ in range(0,256)]
    colors_path = os.path.join(dataset_path, "infos", "colors.txt")
    if os.path.isfile(colors_path):
        colors_ = [line.rstrip('\n').split() for line in open(colors_path)]
        for i in range(0,len(colors_)):
            colors_[i] = [float(c) for c in colors_[i]]
            colors[i] = (colors_[i][0], colors_[i][1], colors_[i][2])
    return colors


def getCategories(dataset_path):
    categories = ["" for _ in range(0,256)]
    categories_path = os.path.join(args.dataset, "infos", "categories.txt")
    if os.path.isfile(categories_path):
        categories = [line.rstrip('\n') for line in open(categories_path)]
    return categories


def getBoxes(boxes_txt_path, format="txt", categories=None):
    if format == "txt":
        boxes = []
        boxes_txt = [line.rstrip('\n').split() for line in open(boxes_txt_path)]
        for box in boxes_txt:
            _box = [int(float(b)) if ix < 5 else float(b) \
                    for ix, b in enumerate(box)]
            boxes.append(_box)
        return boxes
    elif format == "xml":
        tree = ET.parse(boxes_txt_path)
        boxes = []
        for obj in tree.findall('object'):
            _box = [categories.index(obj.find('name').text) + 1, 
                    int(obj.find('bndbox').find('xmin').text),
                    int(obj.find('bndbox').find('ymin').text),
                    int(obj.find('bndbox').find('xmax').text),
                    int(obj.find('bndbox').find('ymax').text)]
            boxes.append(_box)

        return boxes
    else:
        return []


def addBoxToImage(box, image, threshold=None, accepted=None):
    if threshold == None or (threshold != None and box[-1] > threshold):# and box[-1] <= threshold + 0.1):
        cls = int(float(box[0]))
        x1, y1, x2, y2 = int(float(box[1])), int(float(box[2])), int(float(box[3])), int(float(box[4]))
        conf = ' '.join([':', str(box[-1])]) if (threshold != None) else ''

        color = colors[cls-1]
        if accepted != None:
            color = (0,255,0) if accepted else (0,0,255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        point = (x1, y1+15)
        label = categories[cls-1] + conf
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.rectangle(image, (x1, y1), (point[0]+size[0], point[1]+size[1]), (255,255,255), -1)
        cv2.putText(image, label, point, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)


def displayImage():
    curr_categ = _categs_details[_categoryidx]
    curr_detect = _detectionidx[_categoryidx]
    img_name = curr_categ['images_indexes'][curr_detect]
    real_img_ix = _allimages.index(img_name)
    detection = curr_categ['boxes'][curr_detect]
    annotations = _allannotations[real_img_ix]
    tp = curr_categ['true_positives'][curr_detect]
    tpPrev = 0
    if curr_detect != 0:
        tpPrev = curr_categ['true_positives'][curr_detect - 1]
    accepted = (int(float(tp)) - int(float(tpPrev)) > 0)

    image_path = os.path.join(_paths['images'], img_name + "." + args.img_format)
    
    #if not os.path.isfile(image_path):
    #    nextImage(None)
    assert os.path.isfile(image_path), \
           'Error, can\'t find image at {}'.format(image_path)

    title.txt = "Image " + img_name + " - " +  ("CORRECT" if accepted else "INCORRECT")
    title.configure(text = "Image " + img_name)

    # Get the images
    image_ann = cv2.imread(image_path)
    image_det = cv2.imread(image_path)
    
    # Add the boxes
    for box in annotations:
        addBoxToImage(box, image_ann)
    
    # Add the boxes
    #for box in relev_detections:
    #    addBoxToImage(box, image_det, _threshold)
    addBoxToImage(detection, image_det, _threshold, accepted)

    # Convert for TK
    image_ann = cv2.cvtColor(image_ann, cv2.COLOR_BGR2RGBA)
    image_det = cv2.cvtColor(image_det, cv2.COLOR_BGR2RGBA)
    _image_ann = Image.fromarray(image_ann)
    _image_det = Image.fromarray(image_det)
    ratio = min(MAX_WIDTH/float(_image_ann.width), 
                MAX_HEIGHT/float(_image_ann.height))
    new_size = (int(_image_ann.width*ratio), int(_image_ann.height*ratio))
    _image_ann = _image_ann.resize(new_size, Image.ANTIALIAS)
    _image_det = _image_det.resize(new_size, Image.ANTIALIAS)

    tk_image_ann = ImageTk.PhotoImage(image=_image_ann)
    tk_image_det = ImageTk.PhotoImage(image=_image_det)

    # Organize
    display1.imgtk = tk_image_ann
    display1.configure(image = tk_image_ann)
    display2.imgtk = tk_image_det
    display2.configure(image = tk_image_det)


def displayCategory():
    curr_categ = _categs_details[_categoryidx]
    categ_name = categories[_categoryidx]
    ap = curr_categ['ap']
    confs = [-float(x) for x in curr_categ['confidences']]
    tps = curr_categ['true_positives']
    fps = curr_categ['false_positives']
    recs = [float(x) for x in curr_categ['recalls']]
    precs = [float(x) for x in curr_categ['precisions']]

    a.clear()
    b.clear()
    c.clear()
    a.set_ylim(0., 1.)
    b.set_ylim(0., len(recs))
    c.set_ylim(0., 1.)
    a.set_title("Category: " + str(categ_name) + "\tmAP: " + str(ap))
    a.plot(np.arange(0., len(recs), 1.), np.asarray(recs), 'b-', label="REC")
    a.plot(np.arange(0., len(precs), 1.), np.asarray(precs), 'r-', label="PREC")
    b.plot(np.arange(0., len(tps), 1.), np.arange(0., len(tps), 1.), 'g--')
    b.plot(np.arange(0., len(tps), 1.), np.asarray(tps), 'b-')
    c.plot(np.arange(0., len(confs), 1.), np.asarray([0.75]*len(confs)), 'g--')
    c.plot(np.arange(0., len(confs), 1.), np.asarray(confs), 'b-')
    """
    for categ_ix, categ in enumerate(categories):
        curr_tmpcateg = _categs_details[categ_ix]
        confs = [-float(x) for x in curr_tmpcateg['confidences']]
        c.plot(np.arange(0., len(confs), 1.), np.asarray([0.75]*len(confs)), 'g--')
        c.plot(np.arange(0., len(confs), 1.), np.asarray(confs), 'b-')
    """
    c.axvline(_detectionidx[_categoryidx])
    for i in range(curr_categ['number_detections']):
        tp = tps[i]
        tpPrev = 0 if i == 0 else tps[i - 1]
        accepted = (float(tp) - float(tpPrev) > 0)
        if accepted:
            d.axvline(i, color='g')
        else:
            d.axvline(i, color='r')
    d.axis('off')

    legend = a.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    canvas.draw()


def prevCategory(self):
    global _categoryidx
    if _categoryidx > 0:
        _categoryidx -= 1
        displayCategory()
        displayImage()

def nextCategory(self):
    global _categoryidx
    global _detectionidx
    if _categoryidx < len(categories) - 1:
        _categoryidx += 1
        _detectionidx[_categoryidx] = 0
        displayCategory()
        displayImage()

def prevDetect(self):
    global _detectionidx
    if _detectionidx[_categoryidx] > 0:
        _detectionidx[_categoryidx] -= 1
    else:
        _detectionidx[_categoryidx] = _categs_details[_categoryidx]['number_detections'] - 1
    displayCategory()
    displayImage()

def nextDetect(self):
    global _detectionidx
    if _detectionidx[_categoryidx] < _categs_details[_categoryidx]['number_detections'] - 1:
        _detectionidx[_categoryidx] += 1
    else:
        _detectionidx[_categoryidx] = 0
    displayCategory()
    displayImage()

def updateThresh(self):
    global _threshold
    _threshold = slider_thresh.get() / 100.
    displayImage()

def updateDetect(self):
    global _detectionidx
    newDetect = slider_detect.get()
    if newDetect > _categs_details[_categoryidx]['number_detections'] - 1:
        _detectionidx[_categoryidx] = _categs_details[_categoryidx]['number_detections'] - 1
    else:
        _detectionidx[_categoryidx] = newDetect
    displayCategory()
    displayImage()

def exit(self):
    window.destroy()
    raise SystemExit

def generateEmptyDebugDictionary():
    return { 'number_detections': 0, 
             'ap': 0,
             'images_indexes': [],  
             'boxes': [], 
             'confidences': [], 
             'true_positives': [], 
             'false_positives': [], 
             'recalls': [], 
             'precisions': []
            }

    

if __name__ == "__main__":
    # Get the arguments
    args = getArguments()
    
    # Get the colors of the classes
    colors = getColors(args.dataset)
    
    # Get the categories
    categories = getCategories(args.dataset)
    
    # Get the paths
    paths = { 'images': os.path.join(args.dataset, 'Images'),
              'annotations': os.path.join(args.dataset, 'Annotations'),
              'detections': os.path.join(args.results_dir, 'annotations'),
              'details': os.path.join(args.results_dir, 'details') }



    # List all the relevant images
    print 'Get the images...'
    all_images = []
    for img in os.listdir(paths['detections']):
        all_images.append(img.split('.')[0])

    # Get all their annotations
    print 'Get the annotations...'
    all_annotations = []
    for img in all_images:
        boxes = getBoxes(os.path.join(paths['annotations'], img + "." + args.ann_format), args.ann_format, categories)
        all_annotations.append(boxes)

    # Get all their detections
    print 'Get the detections...'
    all_detections = []
    for img in all_images:
        boxes = getBoxes(os.path.join(paths['detections'], img + ".txt"))
        all_detections.append(boxes)


    # Detail per category
    print 'Get the details of each category...'
    categs_details = [generateEmptyDebugDictionary() for _ in range(len(categories))]
    
    # Average precisions
    path = os.path.join(paths['details'], "maps.txt")
    allmaps_splitted = [line.rstrip('\n') for line in open(path)]
    assert (len(allmaps_splitted) == len(categories)), \
        'Error in maps.txt, expects ' + str(len(categories)) + ' values (got ' + len(allmaps_splitted) + ')' 
    for ap_ix, ap in enumerate(allmaps_splitted):
        categs_details[ap_ix]['ap'] = float(ap)

    # Details per annotations
    for categ_ix, categ in enumerate(categories):
        path = os.path.join(paths['details'], "detections_" + str(categ_ix+1) + ".txt")
        if os.path.isfile(path):
            detections = [line.rstrip('\n').split() for line in open(path)]
        else:
            detections = []
        categs_details[categ_ix]['number_detections'] = len(detections) 
        for detection_ix, detection in enumerate(detections):
            categs_details[categ_ix]['images_indexes'].append(detection[0])
            categs_details[categ_ix]['boxes'].append([categ_ix+1, detection[1], detection[2], detection[3], detection[4], -float(detection[5])])
            categs_details[categ_ix]['confidences'].append(detection[5])
            categs_details[categ_ix]['true_positives'].append(detection[6])
            categs_details[categ_ix]['false_positives'].append(detection[7])
            categs_details[categ_ix]['recalls'].append(detection[8])
            categs_details[categ_ix]['precisions'].append(detection[9])

    # Global variables
    global _paths
    global _threshold
    _paths = paths
    _threshold = args.threshold

    global _nbimages
    global _allimages
    global _allannotations
    global _alldetections

    _nbimages = len(all_images)
    _allimages = all_images
    _allannotations = all_annotations
    _alldetections = all_detections

    global _categs_details
    global _detectionidx
    _categs_details = categs_details
    _detectionidx = [0] * len(categories)




    ########################################################
    ########################################################
    for categ_ix, categ in enumerate(categories):
        curr_categ = _categs_details[categ_ix]
        tps = curr_categ['true_positives']
        nbTrue = 0
        nbQuite = 0
        nbFalse = 0
        for i in range(curr_categ['number_detections']):
            tp = tps[i]
            tpPrev = 0 if i == 0 else tps[i - 1]
            diff = float(tp) - float(tpPrev)
            if diff == 1:   nbTrue += 1
            elif diff > 0:  nbQuite += 1
            else:           nbFalse += 1
        print ('\t'.join([str(nbTrue),str(nbQuite),str(nbFalse)]))
    ########################################################
    ########################################################




    

    # TK widget
    window = Tk()
    window.wm_title("Results")
    window.config(background="#FFFFFF", width=MAX_WIDTH*2)
    window.minsize(width=1100, height=900)
    window.maxsize(width=1100, height=900)

    # Graphics
    f = Figure(figsize=(5, 4), dpi=100)
    a = f.add_subplot(411)
    b = f.add_subplot(412)
    c = f.add_subplot(413)
    d = f.add_subplot(414)
    t = np.arange(0.0, 1.0, 1.)
    s = np.asarray([0])
    a.set_title('mAP: 0')
    a.plot(t, s)
    b.plot(t, s)
    c.plot(t, s)

    # Images
    imageFrame = Frame(window, width=MAX_WIDTH*2, height=MAX_HEIGHT)
    title = Label(imageFrame)
    title.pack(side=TOP, fill=BOTH)
    display1 = Label(imageFrame)
    display1.pack(side=LEFT)
    display2 = Label(imageFrame)
    display2.pack(side=RIGHT)
    slider_detect = Scale(window, from_=0, to=500, command=updateDetect, orient=HORIZONTAL, length=500)
    slider_detect.set(int(_threshold * 100))
    slider_detect.pack(side=BOTTOM)

    # Canvas for the graphics
    canvas = FigureCanvasTkAgg(f, master=window)
    canvas.show()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH)
    toolbar = NavigationToolbar2TkAgg(canvas, window)
    toolbar.update()
    canvas._tkcanvas.pack(side=TOP, fill=BOTH)
    imageFrame.pack()

    displayCategory ()
    displayImage()

    # TK event
    window.bind('<Left>', prevCategory)
    window.bind('<Right>', nextCategory)
    window.bind('<Up>', nextDetect)
    window.bind('<Down>', prevDetect)
    window.bind('<Escape>', exit)

    # Loop
    window.mainloop()

    raise SystemExit