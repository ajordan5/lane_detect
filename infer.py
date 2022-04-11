import torch
import segmentation_models_pytorch as smp
import albumentations as albu
import cv2
import matplotlib.pyplot as plt
import numpy as np


"""Load lane detection model and provide all helper functions for lane inference"""

lane_model = torch.load('../datasets/best_model.pth')
ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
out = cv2.VideoWriter('final_lanes.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 60, (640*2,384*2))

# Polygon for lane detection
# Defines a shape that when overlap with lanes is detected, indicates exiting of lane
start = (305,340)
width = 160
pts = np.array([[start[0],start[1]], [start[0]+width,start[1]], [start[0]+width,384], [start[0], 384]], np.int32)
pts = pts.reshape((-1,1,2))
font = cv2.FONT_HERSHEY_SIMPLEX

start = (285+45+23,300)
width = 45
pts2 = np.array([[start[0],start[1]], [start[0]+width,start[1]], [start[0]+width,384], [start[0], 384]], np.int32)
pts2 = pts2.reshape((-1,1,2))

pts3= np.array([pts2[0], pts2[1], pts[2], pts[3]])
pts3 = pts3.reshape((-1,1,2))

if torch.cuda.is_available():
    print("Using local GPU")
    DEVICE = 'cuda'
else:
    print("No GPU found")
    DEVICE = 'cpu'

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def preprocess(image):
    image = cv2.resize(image, (640,384))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=to_tensor),
        ]
    preprocess = albu.Compose(_transform)
    image = preprocess(image=image)["image"]
    img_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    return img_tensor

def visualize(**images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(num=1,figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        # pdb.set_trace()
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    # plt.show()
    plt.pause(0.1)

def video_maker(predict, img, frame, total):
    print("WRITING VIDEO FOR FRAME {}/{}".format(frame, total))
    pr_mask = (predict.squeeze().cpu().numpy().round()).astype(np.uint8)
    pr_mask[np.where(pr_mask==1)] = 255
    mask_inv = cv2.bitwise_not(pr_mask)
    mask_inv_rgb = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
    # img_gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (640,384))
    # display = cv2.bitwise_and(img_gray, img_gray, mask=mask_inv)
    # display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (640,384))
    display = cv2.bitwise_and(img, img, mask=mask_inv)

    # Check for lane deviation
    # cv2.fillConvexPoly(display,pts3,0) # Display lane checking polygon
    display = check_lane(pr_mask, display)
    out.write(display)

def check_lane(mask, image):
    " Check if leaving lane, display warning if necessary, resize for higher resolution text"
    checker = np.zeros_like(mask)
    # cv2.fillConvexPoly(checker,pts,255)
    # cv2.fillConvexPoly(checker,pts2,255)
    cv2.fillConvexPoly(checker,pts3,255)

    overlap = cv2.bitwise_and(checker, mask)
    image = cv2.resize(image, (0,0), fx=2, fy=2) 
    if overlap.any():
        print("overlap!")
        cv2.putText(image, 'WARNING: LANE DEVIATION DETECTED', (30,30), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    return image
