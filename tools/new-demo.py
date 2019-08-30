from torchvision import transforms
from utils_ import *
from PIL import Image, ImageDraw, ImageFont
import torch
import shutil
import subprocess
import glob
from tools.test_ import *
import time
import numpy as np
import torch.multiprocessing as mp
import queue as Queue
from multiprocessing import Pool
import itertools
from multiprocessing.dummy import Pool as ThreadPool 
from pathos.multiprocessing import ProcessingPool  
# import model as modellib  
from imageai.Detection import ObjectDetection
import os
import cv2
from model import *
import argparse
import torchvision.transforms.functional as F
from tools.utils import *
from os.path import isfile, join
from tempfile import TemporaryFile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = '/home/fengy/Documents/SiamMask/BEST_checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint)



#create color array
colors=[]
for i in range(len(rev_label_map)):
    colors.append((255-5*i,5*i,115+5*i))

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])




parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='/home/fengy/Documents/SiamMask/data/video/bus.mp4', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()


def detectBySSD(frame,min_score=0.30,original=False):
    Bounding,labels=detect(F.to_pil_image(frame), min_score=min_score, max_overlap=0.5, top_k=200)
    Bounding=Bounding.detach().numpy() 
    ROIs1=np.array(Bounding)
#    ROIs1=np.array([Bounding[0,0],Bounding[0,1],Bounding[0,2]-Bounding[0,0],Bounding[0,3]-Bounding[0,1]]).astype(int)
    if not original:
        ROIs1=np.zeros((Bounding.shape[0],4))
        for index,b in enumerate(Bounding):
            x,y,x1,y1=b
            w,h=x1-x,y1-y
            ROIs1[index,:]=np.array([int(x),int(y),int(w),int(h)])
    ROIs=ROIs1.astype(int)
    return ROIs, labels



def _extract_frames(v_path):
        '''Extract all fromes from @v_path

        :param v_path: full path to video
        :return: list of full paths to extracted video frames
        '''
        print("Extracting frames from {}".format(v_path))
        # Store frames in tmp dir by default
        tmp_dir = os.path.join('/home/fengy/Documents/SiamMask/data', os.path.basename(v_path))
        # If path exists, delete it
        if os.path.isdir(tmp_dir):
            return tmp_dir
        os.mkdir(tmp_dir)
        
        # TODO Make fps configurable at command line
        cmd = "ffmpeg -i %s -vf fps=55 %s" % (v_path,os.path.join(tmp_dir,'%09d.png'))
        # execute ffmpeg cmd
        subprocess.call(cmd,shell=True)
        return tmp_dir


def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    det_labels = [l for l in det_labels[0].to('cpu').tolist()]
    
    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['__background__']:

        return 

    return det_boxes,det_labels

def saveSSD(ims):
    ssd_dets=[]
    ssd_dets1=[]
    for f, frame in enumerate(ims):
        ROIs,labels=detectBySSD(frame,0.45,True)
        ROIs1,labels1=detectBySSD(frame,0.45)
        ssd_det=np.hstack((ROIs,np.array(labels).reshape(-1,1)))
        ssd_dets.append(ssd_det)
        ssd_det1=np.hstack((ROIs1,np.array(labels1).reshape(-1,1)))
        ssd_dets1.append(ssd_det1)
    np.save(base_path, np.array(ssd_dets))
    np.save(base_path+'roi', np.array(ssd_dets1))
    return 



def convert_frames_to_video(pathIn, fps):
    print("Converting...")

    # define save dir
    pathOut=os.path.join(pathIn, 'bus.avi')
    
    frame_array = []
    files = sorted(glob.glob(join(pathIn, '*.png*')))
 
    for i in range(len(files)):
        filename=join(pathIn,files[i])
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
 
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()



def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou




if __name__ == '__main__':
    # Setup device
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['best_loss']
    print('\nLoaded checkpoint f    output_loc="/home/fengy/Documents/SiamMask/data/bus.mp4_processed"    
    convert_frames_to_video(output_loc,10)rom epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
    model = checkpoint['model']
    model = model.to(device)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    # Setup Model
    cfg = load_config(args)
    from experiments.siammask_sharp.custom_ import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    # Parse Image fil
    base_path=_extract_frames(args.base_path)
    output_loc=base_path+"_processed"
    if not os.path.isdir(output_loc):
        os.mkdir(output_loc)
    img_files = sorted(glob.glob(join(base_path, '*.png*')))
    ims = [cv2.imread(imf) for imf in img_files]
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)

# =============================================================================
#     # Select ROI
#     
#     ROIs = cv2.selectROIs('SiamMask', ims[0], False, False)
# =============================================================================

    
    targets = []
    toc=0
    labels=[]
    ssd_dets=[]
    
    if not isfile(base_path+'.npy')  :
        saveSSD(ims)
    ssd_dets=np.load(base_path+'.npy')
    ssd_dets_roi=np.load(base_path+'roi'+'.npy')

    
    
    
    for f, frame in enumerate(ims):
  # Capture frame-by-frame
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      tic = cv2.getTickCount()
      re_init=False
      

# =============================================================================
#           ROIs,labels=detectBySSD(frame)
#           if len(ROIs)!=len(targets):
#               re_init=True
# =============================================================================
      
      if f == 0 or f%90==0:  # init
          targets = []
          ROIs,labels=detectBySSD(frame,0.4)
          for index, i in enumerate(ROIs):
                x,y,w,h= i
                target_pos = np.array([x  + w / 2, y + h / 2])
                target_sz = np.array([w, h])
                label=labels[index]
                s ={"target_pos":target_pos,"target_sz":target_sz,"x":x,"y":y,"w":w,"h":h,"label":label}
                targets.append(s)
        
          for i in targets:
                print(i["target_pos"])
                print(i["target_sz"])
                
            # state = siamese_init(frame,tar  siammask, cfg['hp'], device=device,targets=targets)  # init tracker
            # state1 = siamese_init(frame, target_pos1, target_sz1, siammask, cfg['hp'], device=device)  # init tracker
          state = siamese_init(frame, siammask, cfg['hp'], device=device,targets=targets)       
          
      elif f > 0 and f%30==0:
          targets=state["targets"]
          ssd_det=ssd_dets[f]
          ssd_det_roi=ssd_dets_roi[f]
          ious=[[]]*len(ssd_det)
          for det_id,det in enumerate(ssd_det):
              ssd_label=det[-1]
              ssd_bounding=det[:-1]
              ious[det_id]=[[]]*len(targets)
              
              for i,target in enumerate(targets):
                  label=target['label']
                  if label==ssd_label:
                      bounding_low=target['target_pos']-target['target_sz']/2
                      bounding_up=target['target_pos']+target['target_sz']/2
                      bounding=np.append(bounding_low,bounding_up)
                      iou=bb_intersection_over_union(ssd_bounding, bounding)
                  else:
                      iou=0.0
                  ious[det_id][i]=iou
          
          ious=np.array(ious)
          for tar_i in reversed(range(len(targets))):
                if max(ious[:,tar_i])<0.1:
                    del targets[tar_i]
            
            
            
          for det_i in range(len(ssd_det)):
                if max(ious[det_i,:])<0.3:
                    x,y,w,h,label= ssd_det_roi[det_i]
                    target_pos = np.array([x  + w / 2, y + h / 2])
                    target_sz = np.array([w, h])
                    s ={"target_pos":target_pos,"target_sz":target_sz,"x":x,"y":y,"w":w,"h":h,"label":label}
                    targets.append(s)
          state = siamese_init(frame, siammask, cfg['hp'], device=device,targets=targets)       


      state = siamese_track(state, frame)
                    
      

      for i,target in enumerate(state["targets"]):
                
          location = target['ploygon'].flatten()
          label=target['label']
          mask = target['mask'] > state['p'].seg_thr
          masks = (mask > 0) * 255     
          masks = masks.astype(np.uint8)
          frame[:, :, 2] = (mask > 0) * 255 + (mask == 0) * frame[:, :, 2]
          cv2.polylines(frame, [np.int0(location).reshape((-1, 1, 2))], True,colors[label], 3)
          font = cv2.FONT_HERSHEY_SIMPLEX
          text_position_x=np.int0(max((location).reshape((-1, 1, 2))[:,0,0])+min((location).reshape((-1, 1, 2))[:,0,0]))//2
          text_position_y=min(np.int0(location).reshape((-1, 1, 2))[:,0,1])
          cv2.putText(frame,rev_label_map[label],(text_position_x,text_position_y), font, 1,colors[label],2,cv2.LINE_AA)     
                
      cv2.imshow('SiamMask', frame)
      cv2.imwrite(join(output_loc,"%#06d.png" % (f)), frame)
      print (time.ctime()) 
      print ("frame",str(f)) 
         

      toc += cv2.getTickCount() - tic
      toc /= cv2.getTickFrequency()
      fps = f / toc
        
        # Display the resulting frame
        # cv2.imshow('Frame',frame)
     
        # Press Q on keyboard to  exit
      if cv2.waitKey(1) & 0xFF == ord('q'):
            break
     

       

     
    # Closes all the frames
    cv2.destroyAllWindows()
    output_loc="/home/fengy/Documents/SiamMask/data/bus.mp4_processed"    
    convert_frames_to_video(output_loc,55)
