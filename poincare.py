#import orientation
import math
import cv2 as cv
import numpy as np

import numpy as np
import math

def find_singularities(orientations, tolerance, background):

    orientations = orientations.astype(np.float32)
    H, W = orientations.shape
    singular_map = np.zeros((H, W), dtype=np.uint8)
    # Define tolerance (in radians) for classifying angle sums
    tolerance = math.radians(tolerance)  
    
    for i in range(2, H-2):
        for j in range(2, W-2):
            neighbors = [
                (i-2, j-2), (i-2, j-1), (i-2, j), (i-2, j+1), (i-2, j+2),    # top row (left to right)
                (i-1, j+2), (i, j+2), (i+1, j+2), (i+2, j+2),                # right column (top to bottom)
                (i+2, j+1), (i+2, j), (i+2, j-1), (i+2, j-2),                # bottom row (right to left)
                (i+1, j-2), (i, j-2), (i-1, j-2)                             # left column (bottom to top)
            ]
           
            sum_diff = 0.0
            
            for k in range(len(neighbors)):
                if neighbors[k] in background or neighbors[(k+1) % len(neighbors)] in background or (i,j) in background:
                    continue
                x1, y1 = neighbors[k]
                x2, y2 = neighbors[(k+1) % len(neighbors)] 
                theta1 = orientations[x1, y1]
                theta2 = orientations[x2, y2]
                d_theta = theta2 - theta1
                
                if d_theta > math.pi/2:
                    d_theta -= math.pi       
                elif d_theta < -math.pi/2:
                    d_theta += math.pi       
                sum_diff += d_theta
            
            if  math.pi - tolerance <= sum_diff <= math.pi + tolerance:
                singular_map[i, j] = 3
            if -math.pi - tolerance <= sum_diff <= -math.pi + tolerance:
                singular_map[i, j] = 4
            if 2*math.pi-tolerance <= sum_diff <= 2*math.pi+tolerance:
                singular_map[i, j] = 5
        
    return singular_map

colors={5: (255, 0, 0), 4: (0, 255, 255), 3: (0, 0, 255)}
def draw_singularities(img, labels, blk):
    out = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    rows,columns = labels.shape
    for r in range(rows):
        for c in range(columns):
            val = int(labels[r, c])
            if val == 0:
                continue
            cv.rectangle(out, (c*blk, r*blk), ((c+1)*blk, (r+1)*blk), colors[val], 2)

    return out
