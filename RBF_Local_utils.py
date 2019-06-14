# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:49:55 2019
@author: mekides Assefa Abebe
@place:NTNU - ColourLab
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import *
import time
from numpy import linalg as LA
import math
from random import random

# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

# Draw a point
def draw_point(img,p, color ) :
    cv2.circle(img,tuple(p.astype(int)), 2, color,-1)
    
# Radial Basis Functions
#-----------------------------------------------------------
#the multiquadric radial basis functions
def phi1(r,r0):
    v = np.zeros(r.shape,np.float32)
    i = r != 0 
    v[i] = r[i] * np.log(r[i])
    return v
def phi2(r,r0):
    v = np.zeros(r.shape,np.float32)
    i = r != 0 
    v[i] = r[i]**2 * np.log( r[i]**2)
    return v

#the gaussian radial basis function
def phi3(r,c):
    v = np.zeros(r.shape,np.float32)
    i = r != 0 
    v[i] = np.exp(-c * r[i]**2)
    return v
#--------------------------------------------------------
# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color) :
    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])
    for t in triangleList :
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
            cv2.line(img, pt1, pt2, delaunay_color, 1)
            cv2.line(img, pt2, pt3, delaunay_color, 1)
            cv2.line(img, pt3, pt1, delaunay_color, 1)
def rand_cluster(n,c,r):
    """returns n random points in disk of radius r centered at c"""
    x,y = c
    points = []
    for i in range(n):
        theta = 2*math.pi*random()
        s = r*random()
        points.append((x+s*math.cos(theta), y+s*math.sin(theta)))
    return points
#RBF interpolation
def my_rbf_interpolation(pts_in,pts_dest,vs_outsal,edge_map,mesh_size):
    '''
    Computes the Radial basis function based interpolation for the retargeted 
    out of salient region vertices
    input:
        pts_in: the in-salient region vertices, including the vertices on the 
        boarders of the image
        pts_dest: the uniformly scaled values of pts_in for the retargeted 
        image location
        vs_outsal: all the vertices found outside of salient as well as boarder 
        regions of the image.
        edge_map: the edge detection map computed from the input image
        mesh_size: the number of pixels in a single mesh element width/height
    output:
        vs_outsal_dest: the interpolated values of vs_outsal 
    '''
    rows,cols = edge_map.shape
    nin = pts_in.shape[0]
    Min = np.zeros((nin,nin),dtype=np.float32)
    nb_edge = np.zeros((1,nin),dtype=np.float32)
    c = cols / mesh_size[0]
    
    #compute average edge map values arround the neughbourhood 
    for j in range(0,nin):
        w_lx = np.maximum(np.minimum(pts_in[j,1],rows -1) - mesh_size[1],0)
        w_rx = np.minimum(np.minimum(pts_in[j,1],rows -1) + mesh_size[1],rows -1)
        w_ly = np.maximum(np.minimum(pts_in[j,0],cols -1) - mesh_size[0],0)
        w_ry = np.minimum(np.minimum(pts_in[j,0],cols -1) + mesh_size[0],cols -1)
        tot_pixels = (w_rx - w_lx) * (w_ry - w_ly)
        nb_edge[0,j] = np.sum(edge_map[w_lx:w_rx,w_ly:w_ry]) / tot_pixels
        
    nb_edge[0,:] = nb_edge[0,:] / max(nb_edge[0,:])
    
#       compute normalizing matrix
#   ---------------------------------------------------------------------------
#       #compute the interpolation kernel
    for i in range(0,nin):
        r = LA.norm (pts_in[i,:] - pts_in,axis=1)#[j,:])#
        Min[i,:] =  np.transpose(nb_edge[0,:] * phi1(r,c))

    #compute normalizing matrix
    gunity = np.ones(np.shape(pts_in),dtype=np.float32)
    alpha_g = np.matmul(LA.pinv(Min),gunity)
    alpha_m = np.matmul(LA.pinv(Min),pts_dest)
    
#    compute the rbf function matrix Mout for the destination vertices
#   ---------------------------------------------------------------------------
    nout = vs_outsal.shape[0];
    Mout = np.zeros((nout,nin),dtype=np.float32)
    
    for i in range(0,nout):
        r = LA.norm (vs_outsal[i,:] - pts_in,axis=1)
        Mout[i,:] = np.transpose(nb_edge[0,:] * phi1(r,c))

    vs_outsal_destf = np.matmul(Mout,alpha_m)
    vs_outsal_destg = np.matmul(Mout,alpha_g)
    vs_outsal_dest = vs_outsal_destf / vs_outsal_destg
    return vs_outsal_dest

def merge_intervals(intervals):
    sorted_by_lower_bound = sorted(intervals, key=lambda tup: tup[0])
    merged = []
    for higher in sorted_by_lower_bound:
        if not merged:
            merged.append(higher)
        else:
            lower = merged[-1]
            # test for intersection between lower and higher:
            # we know via sorting that lower[0] <= higher[0]
            if higher[0] <= lower[1]:
                upper_bound = max(lower[1], higher[1])
                merged[-1] = (lower[0], upper_bound)  # replace by merged interval
            else:
                merged.append(higher)
    return merged
def compute_bbox(contour):
    box = cv2.minAreaRect(contour)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    return box
def find_if_close(cnt1,cnt2,lp):
    box1 = compute_bbox(cnt1)
    box2 = compute_bbox(cnt2)
    dist = np.abs(np.linalg.norm(box1 - box2,axis = 1))
    if any(dist < lp):
        return True
    else:
        return False

#    #combine close regions
#    #--------------------------------------------------------------------------
def merge_close_regions(smap_bw,rows,cols):
    '''
    Merge two roi in to one roi if the distance between them is less than 20% of
    the higher image dimension. 
    input: 
        smap_bw: the binary map of the input image
        rows: height of the input image
        cols: width of the input image
    output: 
        smap_combined: binary map containing the merged roi 
    '''
    smap_combined = smap_bw.copy()#since findContours modify the mask 
    smap2,contours,hier = cv2.findContours(smap_bw,cv2.RETR_EXTERNAL,2)
    LENGTH = len(contours)
    status = np.zeros((LENGTH,1))
    
    for i,cnt1 in enumerate(contours):
        x = i    
        if i != LENGTH-1:
            for j,cnt2 in enumerate(contours[i+1:]):
                x = x+1
                lp = max(rows,cols) * 0.2
                dist = find_if_close(cnt1,cnt2,lp)#check the 20% distance ...
                #difference
                if dist == True:
                    val = min(status[i],status[x])
                    status[x] = status[i] = val
                else:
                    if status[x]==status[i]:
                        status[x] = i+1
    
    unified = []
    maximum = int(status.max())+1
    for i in range(maximum):
        pos = np.where(status==i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i] for i in pos)
            hull = cv2.convexHull(cont)
            unified.append(hull)
    cv2.drawContours(smap_combined,unified,-1,255,-1)
    return smap_combined
#    #-------------------------------------------------------------------------
def compute_regions_stats(smap_bw_ctr):
    '''
    This function computes the total width and height of salient areas as well
    as the centroids of each connected salient regions.
    input: binary map
    output:
        num_labels: number of connected regions 
        labels: region labels
        centroids: centroids of each connected salient regions
        x_length: the total width span of the salient regions
        y_length: the total height span of the salient regions
    '''
    connectivity = 8  
    output = cv2.connectedComponentsWithStats(smap_bw_ctr, connectivity, cv2.CV_32S)
    # Get the results
#    #The number of labels
    num_labels = output[0]
#    #The label matrix
    labels = output[1]
    #The stat matrix
    stats = output[2]
    #width computation
    x_intervals = np.zeros((stats.shape[0],2))
    x_intervals[1:,0] = stats[1:,0]
    x_intervals[1:,1] = stats[1:,0] + stats[1:,2]
    #merge overlapping regions' widths
    x_intervals_merged = np.array(merge_intervals(x_intervals[1:,:]),dtype=float)
    #height computation
    y_intervals = np.zeros((stats.shape[0],2))
    y_intervals[1:,0] = stats[1:,1]
    y_intervals[1:,1] = stats[1:,1] + stats[1:,3]
    #merge overlapping regions' widths
    y_intervals_merged = np.array(merge_intervals(y_intervals[1:,:]),dtype=float)
    #add the widths and heights 
    x_length = np.sum(x_intervals_merged[:,1] - x_intervals_merged[:,0])
    y_length = np.sum(y_intervals_merged[:,1] - y_intervals_merged[:,0])
    # The fourth cell is the centroid matrix
    centroids = output[3]
    
#    centroids = centroids.reshape((centroids.shape[0],1, 2))
    return num_labels,labels,centroids,x_length,y_length 

def correct_for_boundary(pts_in,pts_dest,w,h,wd,hd):
    pts_dest_new = pts_dest.copy()
    #check if the salient region have points on the boarders of the image
    mask_bt = pts_in[:,1] <= 1
    mask_bl = pts_in[:,0] <= 1
    mask_bb = pts_in[:,1] >= (h - 1)
    mask_br = pts_in[:,0] >= (w - 1)
    
    if any(mask_bt):
        pts_dest_new[:,1] = pts_dest[:,1] - np.min(pts_dest[:,1])  
    if any(mask_bl):
        pts_dest_new[:,0] = pts_dest[:,0] - np.min(pts_dest[:,0]) 
    if any(mask_bb):
        pts_dest_new[:,1] = pts_dest[:,1] + (hd - np.max(pts_dest[:,1]))  
    if any(mask_br):
        pts_dest_new[:,0] = pts_dest[:,0] + (wd - np.max(pts_dest[:,0])) 
        
    return pts_dest_new
