#import math
import cv2 as cv
import numpy as np
#from graph import graph, node
class create_graph:
        def __init__(self, singularitie, terminations, bifurcations, slopes, background, kernel_size):
            self.singularities = singularitie
            self.blk = kernel_size
            self.rows, self.cols = len(slopes), len(slopes[0])
            self.minutiaes = self.parse_minutiaes(terminations, bifurcations)
            self.background = background
            self.slopes = slopes
            
            self.node = {1: (255, 0, 0), 2: (0, 255, 255), 3: (0, 0, 255), 4:(255,255,0), 5:(255,0,255)}  # red=ending, Cyan=bifurcation, blue=loop, yellow=delta, magenta=whorl
     
        """
            Things to do:

            don't label minutiae at the edge of the image
        """
        def parse_minutiaes(self, FeaturesTerm, FeaturesBif):
            nrows, ncols = self.rows, self.cols
            minutiaes = np.zeros((nrows, ncols), dtype=np.uint8)

            for f in FeaturesTerm:
                r, c = f.locX // self.blk, f.locY // self.blk
                if 0 <= r < nrows and 0 <= c < ncols:
                    minutiaes[r][c] = 1

            for f in FeaturesBif:
                r, c = f.locX // self.blk, f.locY // self.blk
                if 0 <= r < nrows and 0 <= c < ncols:
                    minutiaes[r][c] = 2

            return minutiaes

        def avg(self, points):
            mean_x = sum(x for x, y in points) // len(points)
            mean_y = sum(y for x, y in points) // len(points)
            centroid = (mean_x, mean_y)
            def dist2(pt):
                return (pt[0] - centroid[0])**2 + (pt[1] - centroid[1])**2

            return min(points, key=dist2)
        
        
        def bfs(self, i, j, feature_map):
            rows, cols = len(feature_map), len(feature_map[0])
            feature = feature_map[i][j]

            horizon = [(i, j)]
            visited = set()          
            group   = set()     

            while horizon:
                x, y = horizon.pop(0)

                if not (0 <= x < rows and 0 <= y < cols) or (x, y) in visited:
                    continue
                visited.add((x, y))

                if feature_map[x][y] != feature:
                    continue

                group.add((x, y)) 

                for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
                    horizon.append((x + dx, y + dy))

            if not group:         
                return feature_map

            keep = self.avg(group)      
            
            for x, y in group:
                if (x, y) != keep:
                    feature_map[x][y] = 0

            return feature_map

            
        def feature_map(self, minutiaes, singularities):
            H, W = len(singularities), len(singularities[0])
            features_map = [[0 for _ in range(W)] for _ in range(H)]
            for row in range(H):
                for col in range(W):
                    s_val=singularities[row][col]
                    m_val=minutiaes[row][col]
                    features_map[row][col]=max(s_val,m_val)
            return features_map
        
        def clean_features(self,feature_map):
            rows,cols=len(feature_map), len(feature_map[0])
            for r in range(rows):
                for c in range(cols):
                    if feature_map[r][c]!=0:
                        self.bfs(r,c,feature_map)
                    if (r,c) in self.background:
                        feature_map[r][c]=0
            
            return feature_map
        
        
        def draw_singularitiess(self, img, labels, blk):
            out = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            rows, cols = len(labels), len(labels[0])
            for r in range(2,rows-2):
                for c in range(2,cols-2):
                    val = int(labels[r][c])
                    if val == 0:
                        continue
                    cv.rectangle(out, (c*blk, r*blk), ((c+1)*blk, (r+1)*blk), self.node[val], 2)

            return out

            