# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:49:55 2019
@author: mekides Assefa Abebe
@place:NTNU - ColourLab
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
#from scipy import *
#from scipy.interpolate import Rbf
#from numpy import linalg as LA
import numpy.matlib
import time
from os import listdir
from os.path import isfile, join
from RBF_Local_utils import *
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.filters import threshold_otsu
    
def main_RBF(img,r,smap,strength,Es):
    """
    This is the main function which will compute the RBF interpolation of non- 
    salient regions.
    Inputs:
        img: the input image
        r: the scaling factor. destination size = size of img * r
        smap: binary map of salient regions of ROI
        strength: the strength of the compression on the salient regions
                it is denoated as epislon in the corresponding paper
        Es: is the width/height of a single mesh element in number of pixels
    Output:
        img_out: retargeted image
    """
    start = time.time()
        
     #Generate mesh
    rows,cols,ch = img.shape
    ratioh = r[0]
    ratiow = r[1]
    rowsd = int(rows * ratioh)#target image width
    colsd = int(cols * ratiow)#target image height
    mesh_sizex = int(cols/Es)
    mesh_sizey = int(rows/Es)
    x = np.linspace(0,cols,mesh_sizex)
    y = np.linspace(0,rows,mesh_sizey)
    xv, yv = meshgrid(x, y, sparse=False, indexing='ij')
    #all nodes for the input image mesh
    pts = np.zeros((mesh_sizex * mesh_sizey,2))
    pts[:,0] = xv.reshape((mesh_sizex * mesh_sizey))
    pts[:,1] = yv.reshape((mesh_sizex * mesh_sizey))
#    
#    ##assuming we have binary mask, combine close regions
    #for more structured RBF input and avoiding overlaping
#    #-------------------------------------------------------------------------
    smap_bw = smap
    smap_combined = merge_close_regions(smap_bw,rows,cols)
#    cv2.imshow('Binary mask1',smap_combined)
#    #-------------------------------------------------------------------------

#   Changing the connected components to bounding boxes 
#   ---------------------------------------------------------------------------
    connectivity = 8  
    # Perform the operation
    output = cv2.connectedComponentsWithStats(smap_combined,\
                                              connectivity,\
                                              cv2.CV_32S)
    stats = output[2]
    for rgns in range(1,stats.shape[0]):
        hl = max(stats[rgns,1] - mesh_sizey,0)
        hr = min(stats[rgns,1] + stats[rgns,3] + mesh_sizey,rows)
        wl = max(stats[rgns,0] - mesh_sizex,0)
        wr = min(stats[rgns,0] + + stats[rgns,2] + mesh_sizex,cols)
        smap_combined[hl:hr,wl:wr] = 255;
#   ---------------------------------------------------------------------------
    
    smap_bw_ctr = smap_combined.copy()
#    cv2.imshow('Binary mask2',smap_bw_ctr)
#   compute the total length and height of the connected components and their
#   centers
#   ---------------------------------------------------------------------------
    num_labels,labels,centroids,\
    x_length,y_length = compute_regions_stats(smap_bw_ctr)
#    --------------------------------------------------------------------------
    
#   ---------------------------------------------------------------------------
#   classfication of in and out-of-salient regions and pre-processing of 
#   in-salient and boarder vertices
#   ---------------------------------------------------------------------------
    #choose vertices in the salient regions and scale them
    vs_insal = np.zeros((1,2),dtype=np.int32) #
    vs_outsal = np.zeros((1,2),dtype=np.int32) #
    vs_insal_dest = np.zeros((1,2),dtype=np.int32) #
    
    new_centroids = np.zeros(centroids.shape,dtype=int)
    pts_in_mark = np.zeros(pts.shape[0],dtype=int)
    new_r = np.zeros(centroids.shape,dtype=float)
    if num_labels > 0:
        for i in range(1,num_labels):
            mask = np.zeros(smap_bw.shape,np.uint8)
            mask[labels == i] = 255

#            cv2.imshow('mask' + str(i),mask)
                
            # computing centroid scaling constant based on the deformation
            # strength and available space in the destination image (colsd,rowsd)
            Rcols = (cols - x_length) * (1 - strength)
            Rrows = (rows - y_length) * (1 - strength)
            if (x_length + Rcols) < colsd:
                new_r[i,0] = 1
            else:
                new_r[i,0] = colsd /(x_length + Rcols)
            if (y_length + Rrows) < rowsd:
               new_r[i,1] = 1
            else:
                new_r[i,1] = rowsd/(y_length + Rrows)
                
                
            vs_insal_temp = np.zeros((1,2),dtype=np.int32)
            vs_insal_dest_temp = []

            pts_lr = np.minimum(np.fliplr(pts.astype("int")),\
                                       np.array([rows-1,cols-1]))
            pts_idx = mask[pts_lr[:,0],pts_lr[:,1]]
            vs_insal_temp = np.concatenate((vs_insal_temp,\
                                            pts[pts_idx > 0]),\
                                            axis=0)
            pts_in_mark[pts_idx > 0] = 1
            
            vs_insal = np.concatenate((vs_insal,vs_insal_temp[1:,:]), axis=0)
            new_centroids = centroids[i] * np.array([ratiow,ratioh])
            vs_insal_dest_temp = np.subtract(vs_insal_temp[1:,:],centroids[i])\
                                * new_r[i] + new_centroids
            #Adjust for boundary conditions 
            vs_insal_dest_temp = correct_for_boundary(vs_insal_temp[1:,:],\
                                                      vs_insal_dest_temp,\
                                                      cols,rows,colsd,rowsd)
            
            vs_insal_dest = np.concatenate((vs_insal_dest,vs_insal_dest_temp),\
                                           axis=0)

        vs_insal = vs_insal[1:,:]
        vs_insal_dest = vs_insal_dest[1:,:]
        vs_outsal = pts[pts_in_mark < 1]
        
        #identify boundary vertices and remove them from out of salient list
        mask1 = vs_outsal[:,1] == 0
        bt = vs_outsal[mask1,:]
        vs_outsal = vs_outsal[np.logical_not(mask1),:]
        mask2 = vs_outsal[:,0] == 0
        bl = vs_outsal[mask2,:]
        vs_outsal = vs_outsal[np.logical_not(mask2),:]
        mask3 = vs_outsal[:,1] == rows
        bb = vs_outsal[mask3,:]
        vs_outsal = vs_outsal[np.logical_not(mask3),:]
        mask4 = vs_outsal[:,0] == cols
        br = vs_outsal[mask4,:]
        vs_outsal = vs_outsal[np.logical_not(mask4),:]
        
        #Retarget and combine boundary vertices
        bnd_vs = np.concatenate((bt,bl,bb,br), axis=0)
        bnd_vs_dest = bnd_vs * [[ratiow,ratioh]]
        pts_in = np.concatenate((bnd_vs,vs_insal), axis=0) 
        pts_dest = np.concatenate((bnd_vs_dest,vs_insal_dest), axis=0)
    
    vs_outsal = np.array(vs_outsal,dtype=np.int32) 
    pts_in = np.array(pts_in,dtype=np.int32)
    vs_insal = np.array(vs_insal,dtype=np.int32)
    pts_dest = np.array(pts_dest,dtype=np.int32) 
    vs_insal_dest = np.array(vs_insal_dest,dtype=np.int32)      
    
    end = time.time()
    print("preprocessing done in: %.4f time" % (end - start))
#   ---------------------------------------------------------------------------
#   compute the rbf interpolation of the out of salient region vertices 
#   ---------------------------------------------------------------------------
    startrbf = time.time()
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = np.uint8(np.absolute(cv2.Laplacian(gray_image,cv2.CV_64F))) / 256
    vs_outsal_dest = my_rbf_interpolation(pts_in,pts_dest,vs_outsal,laplacian,\
                                          [mesh_sizex,mesh_sizey])
    endrbf = time.time()
    print("RBF interpolation done in: %.4f time" % (endrbf - startrbf))
    
    vs_outsal_dest = np.array(vs_outsal_dest,dtype=np.int32) 
    
##sorting the new pixel indexes and converting back to mesh for correction 
#------------------------------------------------------------------------------
    pts_all_in = np.concatenate((pts_in,vs_outsal), axis=0)
    pts_all_dest = np.concatenate((pts_dest,vs_outsal_dest), axis=0)
    pts_in_final = np.zeros(pts.shape,dtype=np.int32)
    pts_dest_final = np.zeros(pts.shape,dtype=np.int32)
    for j in range(0,pts.shape[0]):
        idx1 = np.logical_and(np.absolute(pts_all_in[:,0] - pts[j,0]) < 1,\
                              np.absolute(pts_all_in[:,1] - pts[j,1]) < 1)
        pts_in_final[j,:] = np.average(pts_all_in[idx1,:], axis=0)
        pts_dest_final[j,:] = np.average(pts_all_dest[idx1,:], axis=0)
    
    pts_dest_meshx =  pts_dest_final[:,0].reshape((mesh_sizex , mesh_sizey))
    pts_dest_meshy =  pts_dest_final[:,1].reshape((mesh_sizex , mesh_sizey))
    
    pts_dest_meshx_median = np.zeros((mesh_sizex , 1),dtype=np.int32)
    pts_dest_meshy_median = np.zeros((1,mesh_sizey),dtype=np.int32)
    pts_dest_meshx_median[:,0] = np.median(pts_dest_meshx, axis=1)
    pts_dest_meshx_str = np.matlib.repmat(pts_dest_meshx_median,1,mesh_sizey)
    pts_dest_meshy_median[0,:] = np.median(pts_dest_meshy, axis=0)
    pts_dest_meshy_str = np.matlib.repmat(pts_dest_meshy_median,mesh_sizex,1)
    
    pts_dest_final[:,0] = pts_dest_meshx_str.reshape((mesh_sizex * mesh_sizey))
    pts_dest_final[:,1] = pts_dest_meshy_str.reshape((mesh_sizex * mesh_sizey))
    
    start = time.time()
#------------------------------------------------------------------------------
#delenouy triangulation and image warping
#------------------------------------------------------------------------------
    img_dest = np.zeros((int(rows * ratioh),int(cols * ratiow),3),np.uint8)
    img_dest[:] = 255
    pts_dest_final[pts_dest_final < 0] = 0
    pts_in_final[pts_in_final < 0] = 0

    # Define colors for drawing.
    delaunay_color = (255, 0, 255)
    # Define the space you want to partition using a rectangle
    rect = (0, 0, cols+1, rows+1)   
    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect);
    
     # Insert points into subdiv
    for p in pts_in_final:
        subdiv.insert((min(p[0,],cols),min(p[1,],rows)))
        
    # Draw delaunay triangles
    img_copy = img.copy()
    # Draw delaunay triangles
    draw_delaunay(img_copy, subdiv, (255, 255, 255) );
#    cv2.imshow(win_delaunay, img_copy)
    
    #get the triangles 
    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])
    img_out = np.zeros((rowsd,colsd,3), dtype = img.dtype)
    # Output image is set to white
#    trngls = 0;
    for t in triangleList :
         
        pt1_in = np.array([t[0], t[1]], dtype=np.int32)
        pt2_in = np.array([t[2], t[3]], dtype=np.int32)
        pt3_in = np.array([t[4], t[5]], dtype=np.int32)
        is_salient1 = vs_insal == pt1_in
        is_salient2 = vs_insal == pt2_in
        is_salient3 = vs_insal == pt3_in
        if np.any(np.logical_and(is_salient1[:,0],is_salient1[:,1]))\
           and np.any(np.logical_and(is_salient2[:,0],is_salient2[:,1]))\
           and np.any(np.logical_and(is_salient3[:,0],is_salient3[:,1])):
             delaunay_color = (255,0,255)
        else:
            delaunay_color = (255,0,0)
#        print('triangles:',pt1_in,pt2_in,pt3_in)
        if rect_contains(r, pt1_in) and rect_contains(r, pt2_in)\
           and rect_contains(r, pt3_in) :
            pts1s = pts_in_final == pt1_in
            pts1s = np.logical_and(pts1s[:,0],pts1s[:,1])
            pts1_out = pts_dest_final[pts1s,:]
            pts2s = pts_in_final == pt2_in
            pts2s = np.logical_and(pts2s[:,0],pts2s[:,1])
            pts2_out = pts_dest_final[pts2s,:]
            pts3s = pts_in_final == pt3_in
            pts3s = np.logical_and(pts3s[:,0],pts3s[:,1])
            pts3_out = pts_dest_final[pts3s,:]
            
            cv2.line(img_dest,(pts1_out[0][0],pts1_out[0][1]),\
                     (pts2_out[0][0],pts2_out[0][1]), delaunay_color,1)
            cv2.line(img_dest, (pts2_out[0][0],pts2_out[0][1]),\
                     (pts3_out[0][0],pts3_out[0][1]), delaunay_color,1)
            cv2.line(img_dest, (pts3_out[0][0],pts3_out[0][1]),\
                     (pts1_out[0][0],pts1_out[0][1]), delaunay_color,1)
            
            #----------------------------------------------------------------------
            #Triangular warping
            #----------------------------------------------------------------------
            # Define input and output triangles 
            tri1 = np.float32([[pt1_in, pt2_in,pt3_in]])
            tri2 = np.float32([[pts1_out[0],pts2_out[0],pts3_out[0]]])
    #        print(tri2)
            # Find bounding box. 
            r1 = cv2.boundingRect(tri1)
            r2 = cv2.boundingRect(tri2)
            # Offset points by left top corner of the 
            # respective rectangles
            tri1Cropped = []
            tri2Cropped = []
                 
            for i in range(0, 3):
              tri1Cropped.append(((tri1[0][i][0] - r1[0]),\
                                  (tri1[0][i][1] - r1[1])))
              tri2Cropped.append(((tri2[0][i][0] - r2[0]),\
                                  (tri2[0][i][1] - r2[1])))
             
            # Apply warpImage to small rectangular patches
            img1Cropped = img[r1[1]:min(r1[1] + r1[3],rows),\
                              r1[0]:min(r1[0] + r1[2],cols)]
            # Given a pair of triangles, find the affine transform.
            warpMat = cv2.getAffineTransform( np.float32(tri1Cropped),\
                                             np.float32(tri2Cropped) )
            # Apply the Affine Transform just found to the src image
            img2Cropped = cv2.warpAffine( img1Cropped, warpMat,\
                                         (min(r2[2],colsd), min(r2[3],rowsd)),\
                                         None, flags=cv2.INTER_LINEAR,\
                                         borderMode=cv2.BORDER_REFLECT_101 )
            # Get mask by filling triangle
            mask = np.zeros((min(r2[3],rowsd), min(r2[2],colsd), 3),\
                            dtype = np.float32)
            cv2.fillConvexPoly(mask, np.int32(tri2Cropped),\
                               (1.0, 1.0, 1.0), 16, 0);
             
            # Apply mask to cropped region
            img2Cropped = img2Cropped * mask
             
            # Copy triangular region of the rectangular patch to the output image
            crop_shape = img_out[r2[1]:min(r2[1]+r2[3],rowsd),\
                                 r2[0]:min(r2[0]+r2[2],colsd)].shape
            if crop_shape[0] != 0 and crop_shape[1] != 0:
                img_out[r2[1]:min(r2[1]+r2[3],rowsd),r2[0]:min(r2[0]+r2[2],colsd)]\
                = img_out[r2[1]:min(r2[1]+r2[3],rowsd),r2[0]:min(r2[0]+r2[2],colsd)]\
                * ( (1.0, 1.0, 1.0) - mask[0:min(r2[1]+r2[3],rowsd) - r2[1],\
                   0:min(r2[0]+r2[2],colsd)-r2[0]])
                     
                img_out[r2[1]:min(r2[1]+r2[3],rowsd),r2[0]:min(r2[0]+r2[2],colsd)]\
                = img_out[r2[1]:min(r2[1]+r2[3],rowsd),r2[0]:min(r2[0]+r2[2],colsd)]\
                + img2Cropped[0:min(r2[1]+r2[3],rowsd) - r2[1],\
                              0:min(r2[0]+r2[2],colsd)-r2[0]]
    end = time.time()
    print("Image warping done in: %.4f time" % (end - start))
    #--------------------------------------------------------------------------
    return img_out, img_dest