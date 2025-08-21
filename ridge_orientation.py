import os, cv2, math, numpy as np
#import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
#import collections
import numpy as np

"""
Things to do:

cite where the algorithms in the functions came from
explain what each function does, parameters, and return values

"""

def block_orientation(img, blk):
    """Eq (5‑7): raw least‑squares orientation per blk×blk block."""
    background = set()
    foreground = set()
    
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    h, w = img.shape
    nrows, ncols = h // blk, w // blk
    slope = np.zeros((nrows, ncols), np.float32)
    fg = np.zeros((nrows, ncols), np.float32)

    for r in range(nrows):
        for c in range(ncols):
            r0, c0 = r*blk, c*blk
            
            #block  = img[r0:r0+blk, c0:c0+blk]

            gxs = gx[r0:r0+blk, c0:c0+blk].ravel()
            gys = gy[r0:r0+blk, c0:c0+blk].ravel()
            
            vxx = np.sum(gxs**2 - gys**2)
            vxy = 2*np.sum(gxs*gys)
            
            θ = 0.5 * math.atan2(vxy, vxx)     # Eq 7 
            if θ == 0:
                background.add((r,c))
            else:
                foreground.add((r,c))
                fg[r][c]=1
            slope[r, c] = θ
    
    return slope, background, foreground, fg


def smooth_orientation(Ori, sigma=2):
    """Eq (8‑12): convert → vector field, low‑pass filter, back‑convert."""
    fx, fy = np.cos(2*Ori), np.sin(2*Ori)
    fx = gaussian_filter(fx, sigma=sigma, mode='nearest')
    fy = gaussian_filter(fy, sigma=sigma, mode='nearest')
    return 0.5 * np.arctan2(fy, fx)               # Eq 12


def draw_field(img, Ori, blk, foreground, scale=4, colour=(0,255,0)):
    out = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    #figure out how to turn background into a set
    for r in range(len(foreground)):
        for c in range(len(foreground[0])):
            if foreground[r][c]==0:
               continue    
            θ = Ori[r, c]
            x0, y0 = int((c+0.5)*blk), int((r+0.5)*blk)
            dx, dy = scale*math.cos(θ+math.pi/2), scale*math.sin(θ+math.pi/2)
            pt1 = (int(x0-dx), int(y0-dy))
            pt2 = (int(x0+dx), int(y0+dy))
            
            cv2.line(out, pt1, pt2, colour, 1, cv2.LINE_AA)

    return out
