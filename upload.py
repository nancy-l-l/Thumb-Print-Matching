import os, cv2, glob, math, numpy as np
import matplotlib.pyplot as plt
#from scipy.ndimage import gaussian_filter
#import matplotlib.image as mpimg
import clean
import ridge_orientation
#import imageio.v3 as iio
#from PIL import Image   
#from scipy.spatial import cKDTree
#from skimage.morphology import (
#    skeletonize, remove_small_objects, disk, closing, dilation
#)
from scipy.ndimage import rotate
#from crossing_number import calculate_minutiae, draw_minutiae
from poincare import find_singularities, draw_singularities
from create_graphs import create_graph
#from augment import rotate
import random
import graph
import fingerprint_feature_extractor

BLK = 8
tolerance = 20  

"""
Things to do:

don't label minutiae at the edge of the image
discard smaller bands of singularities


"""   

def draw_fg(img, blk, fg, colour=(0,255,0)):
    out = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
   
    for r in range(len(fg)):
        for c in range(len(fg[0])):
            if fg[r][c]==1:
                y0, x0 = int((r+0.5)*blk), int((c+0.5)*blk)
                cv2.circle(out, (x0, y0), 3, colour, 1, cv2.LINE_AA)
    return out

def rotate_slopes(slopes: np.ndarray, delta_deg: float) -> np.ndarray:
    delta_rad = math.radians(delta_deg)
    new_slopes = slopes - delta_rad
    new_slopes = (new_slopes + math.pi/2) % math.pi - math.pi/2
    
    return new_slopes

def fg_to_set(foreground):
    fg = set()
    for r in range(len(foreground)):
        for c in range(len(foreground[0])):
            if foreground[r][c]==1:
                fg.add((r,c))
    return fg

def shows(images):
    n=len(images)
    plt.figure(figsize=(2*n,5))
    for i in range(n):
        plt.subplot(1,n,i+1); plt.imshow(images[i], cmap="gray"); plt.axis("off"); plt.title(i+1)
    plt.show()

def parse(img, random_angle, rotated_img):
    
    cleaned_img = clean.enhance_fingerprint(img, resize=True, ridge_segment_thresh=0.05, # less strict than 0.10
            min_wave_length=3,         # allow slightly tighter ridges
            max_wave_length=20)	
    
    rot_cleaned_img = rotate(cleaned_img,angle=random_angle,reshape=True, order=0,mode='constant', cval=0)
    
    FeaturesTerminations, FeaturesBifurcations, feature_img = fingerprint_feature_extractor.extract_minutiae_features(cleaned_img, spuriousMinutiaeThresh=10, invertImage=False)
    
    rot_feature_img = rotate(feature_img, angle=random_angle, reshape=True)
    
    slopes, background,foreground,fg    = ridge_orientation.block_orientation(img, BLK)
    fg_img=draw_fg(img, BLK, fg, colour=(0,255,0))
   
    smoothed_slope   = ridge_orientation.smooth_orientation(slopes)           # Eq 8‑12
    orientation_img  = ridge_orientation.draw_field(cleaned_img, smoothed_slope, BLK, fg)
    
    singularities = find_singularities(smoothed_slope, tolerance, background)
    singularities_img = draw_singularities(cleaned_img, singularities, BLK)
    
    test = create_graph(singularities, FeaturesTerminations, FeaturesBifurcations, smoothed_slope, background, BLK)
    all_features=test.feature_map(test.minutiaes, singularities)
    
    cleaned_features=test.clean_features(all_features)
    cleaned_ft_img=test.draw_singularitiess(cleaned_img, cleaned_features, BLK)
    
    w,h = len(cleaned_features[0]), len(cleaned_features)
   
    rot_foreground = rotate(fg,angle=random_angle,reshape=True, order=0,mode='constant', cval=0)
    rot_fg_img=draw_fg(rotated_img, BLK, rot_foreground, colour=(0,255,0))
    
    rot_cleaned_features = rotate(cleaned_features, angle=random_angle, reshape=True, order=0, mode='constant', cval=0)
    rot_slope = rotate_slopes(smoothed_slope, random_angle)
    rot_slopes = rotate(rot_slope, angle=random_angle, reshape=True, order=0, mode='constant', cval=0)
    rot_smoothed_slope = ridge_orientation.smooth_orientation(rot_slopes)  
    
    rot_orientation_img  = ridge_orientation.draw_field(rotated_img, rot_smoothed_slope, BLK, rot_foreground)
    rot_ft_img=test.draw_singularitiess(rotated_img, rot_cleaned_features, BLK)
    
    rot_fg_set = fg_to_set(rot_foreground)
    
    
    return rot_cleaned_img, smoothed_slope, rot_smoothed_slope, cleaned_features, rot_cleaned_features, orientation_img, rot_orientation_img, cleaned_ft_img,  rot_ft_img,  foreground, rot_fg_set, singularities_img, feature_img, rot_feature_img

def upload(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)	
    random_angle = random.uniform(0, 180) 
    
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), random_angle, 1.0)

    # compute new bounding size
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # adjust the rotation matrix to shift result to the center
    M[0, 2] += (new_w / 2) - (w / 2)
    M[1, 2] += (new_h / 2) - (h / 2)

    rot_img = cv2.warpAffine(
        img, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )
    cleaned_img = clean.enhance_fingerprint(img, resize=True, ridge_segment_thresh=0.05, 
    min_wave_length=3,        
    max_wave_length=20)		
    
    
    rot_cleaned_img = rotate(cleaned_img, angle=random_angle, reshape=True)
    
    rot_cleaned_img, smoothed_slope, rot_smoothed_slope, cleaned_features, rot_cleaned_features, orientation_img, rot_orientation_img, cleaned_ft_img,  rot_ft_img,  foreground, rot_fg_set, singularities_img, feature_img, rot_feature_img = parse(cleaned_img, random_angle, rot_cleaned_img)
    
    rotated_singularities_img = rotate(singularities_img, angle=random_angle, reshape=True)
    g = graph.build_graph(cleaned_features, foreground, smoothed_slope)
    
    rot_g = graph.build_graph(rot_cleaned_features, rot_fg_set, rot_smoothed_slope)
    
    
    #shows([cleaned_img, cleaned_ft_img])
    
    #shows([img, rot_img, cleaned_img, rot_cleaned_img, orientation_img, rot_orientation_img, feature_img, rot_feature_img, singularities_img, rotated_singularities_img, cleaned_ft_img, rot_ft_img])
    #shows([img, cleaned_img, rotated_img, orientation_img,  rot_orientation_img, cleaned_ft_img, rot_ft_img, singularities_img])
    return g, rot_g, random_angle


if __name__ == "__main__":
    
    ROOT_DIR  = "fingerprints"
    SUBDIRS   = ["DB1_B", "DB2_B", "DB3_B", "DB4_B"]
    
    for db in SUBDIRS:
        dir_path = os.path.join(ROOT_DIR, db)
        if not os.path.isdir(dir_path):
            print(f"⚠️  {dir_path} not found – skipping")
            continue

        for root, _, files in os.walk(dir_path):
            for fname in files:
                _, ext = os.path.splitext(fname)
            
                file_path = os.path.join(root, fname)
                
                g, rot_g, angle = upload(file_path)
                
    
    