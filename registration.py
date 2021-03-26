folder_pre = '<PATH_TO_IMAGE_FOLDER>'

# For Dagmar, bigger EM:
template_beg = [200,200,100]
reged_siz = (400,400,7000)

num_of_segs = 100
seg_siz_in_template = 30

# for morph (straightening) movie
#num_of_segs = 100
#seg_siz_in_template = 44

#template_beg = [20,100,100]
#reged_siz = (40,900,4600)

interpulation_order = 1
alpha = 4

#Confocal z scaling:
CONFOCAL_SCALING = 2.727
#EM scaling:
EM_SCALING = 0.45

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--worm_num')
parser.add_argument('-i', '--im_type')
parser.add_argument('-r', '--run_only')
parser.add_argument('-m', '--morph_step', default="103")
parser.add_argument('-s', '--segments_range')
args = parser.parse_args()
wormnum = args.worm_num
imtype = args.im_type
runonly = args.run_only
morphstep = args.morph_step
segsrange= args.segments_range
print('worm number: ' + wormnum, flush=True)
print('image type: ' + imtype, flush=True)

folder = folder_pre + wormnum + '/' 
im_type = imtype

if wormnum=='EM':
	z_scale = EM_SCALING
elif 'Con' in wormnum:
    z_scale = CONFOCAL_SCALING
else:
    z_scale = 1

print(morphstep, flush=True)
morphstep = (float(morphstep)-3)/100

if segsrange:
    num_of_segs = int(segsrange)
    
############################################################
import numpy as np

import tifffile as tif

from scipy.ndimage.measurements import label
from scipy.ndimage import affine_transform
from scipy.ndimage import filters

from skimage.morphology import skeletonize_3d

import math

import time 
import sys
import os
############################################################

###################### SKELETONIZE #########################

worm_mask = tif.imread(folder + 'mask.tif')
siz = worm_mask.shape

skelet_pix_file = [f for f in os.listdir(folder) if 'skelet_pix.npy' in f]

if skelet_pix_file:
    skelet_pix = np.load(folder + 'skelet_pix.npy')
else:
    rough_mask_gaus = filters.gaussian_filter(worm_mask, 2)
    worm_mask = rough_mask_gaus>200

    t = time.time()

    skelet_raw = skeletonize_3d(worm_mask)

    skelet_raw_c = np.copy(skelet_raw==255)

    # To trim loose edges:

    # Remove all pixels that are not connected to the skeleton:

    # Find all connected objects in the skeleton image:
    ske_labeled, n_obj = label(skelet_raw_c,[[[1]*3]*3]*3)
    # Find how many pixels belong to each component:
    unique_vals_in_labeled, pix_counter_per_obj = np.unique(ske_labeled, return_counts=True)
    # Find the index of the unique value for the skeleton object:
    skelet_ind_in_uniq = np.argsort(pix_counter_per_obj)[-2]
    # Now find the unique value:
    skelet_val_in_labeled = unique_vals_in_labeled[skelet_ind_in_uniq]
    # Now remove all smaller objects from skeleton image:
    skelet_rough = ske_labeled==skelet_val_in_labeled

    # Find all the pixels on the rough skeleton:
    pixs_in_skeleton = np.asarray(np.nonzero(skelet_rough))

    # Then find the beginning and the end by finding the ones closest to the edges of the y axis:
    ind_4_sort = np.argsort(pixs_in_skeleton[2,:])
    skelet_beg = pixs_in_skeleton[:,ind_4_sort[0]]
    skelet_end = pixs_in_skeleton[:,ind_4_sort[-1]]
    pixs_in_skelet = pixs_in_skeleton[:,ind_4_sort].T.tolist()

    # Now find shortest path between the beginning and end points:
    # Using dijkstra

    # Make adjacency matrix:

    siz_skelet = len(pixs_in_skelet)

    #adj_mat = np.full((siz_skelet,27),-1,dtype=np.int)
    adj_mat = []
    for i in range(siz_skelet):
        pix = pixs_in_skelet[i]
        count = 0
        neighs = []
        for j in [-1,0,1]:
            for k in [-1,0,1]:
                for l in [-1,0,1]:
                    adj = [x + y for x, y in zip(pix,[j,k,l])]
                    if adj!=pix and adj in pixs_in_skelet:
                        #adj_mat[i,count]=pixs_in_skelet.index(adj)
                        neighs.append(pixs_in_skelet.index(adj))
                        count += 1
        adj_mat.append(neighs)

    # Now, dijkstra:

    # Init a mat to represet the graphs costs:
    # Each row is a node (each node is a pixel in the skeleton)
    # 1st column is the cost of traveling from the beginning node to it.
    # 2nd column is the previous node that led to this node in the shortest path.
    # This enables us at the end to track back the shortest path.
    dij_mat = np.array([[siz_skelet+2]*siz_skelet,[-1]*siz_skelet]).T

    # cost of traveling from first node to itself is =0
    dij_mat[0,0] = 0

    # I set the cost of traveling to a neighbor as 1 (all edges=1)

    i=0
    while i<siz_skelet:
        
        # The cost to traveling to the neighbors of the corrent node is - cost of the current node + 1
        nodes_cost = dij_mat[i,0] + 1
        
        smallest = i
        for neigh in adj_mat[i]:

            if dij_mat[neigh,0] > nodes_cost:
                dij_mat[neigh,:]=[nodes_cost,i]
                if neigh<smallest:
                    smallest=neigh
        
        i = i+1 if i==smallest else smallest

    # Last step in dijksizstra - tracing the path back:
    skelet = np.zeros(siz, dtype=bool)
    skelet_pix = []
    i=siz_skelet-1
    while i!=0:
        skelet[tuple(pixs_in_skelet[i])]=True
        skelet_pix.append(pixs_in_skelet[i])
        i = dij_mat[i,1]

    skelet_pix = np.flip(np.asarray(skelet_pix), axis=0)

    np.save(folder + 'skelet_pix',skelet_pix)
    tif.imsave(folder + 'skeleton.tif', skelet.astype(np.uint8))
        
    print('xxxxx Finished skeletonize xxxxx Time:' + str(time.time()-t), flush=True)

###################### Set Registration Matrices ######################

# We need translation, rotation, and scaling
# But we can't compute the affine transformation since we need minimum 3 points
# So we compute each seperately, and use homogeneous coordinates to unite them in one matrix.

# homogeneous coordinates - using projective geometry (adding the W dimension)
# in our case W=1

def scaling_mat(l0,l1):

    len0 = np.sqrt(np.sum(np.square(l0[0]-l0[1])))
    len1 = np.sqrt(np.sum(np.square(l1[0]-l1[1])))
    
    s = len0/len1
    return np.array([[s*z_scale,0,0,0],[0,s,0,0],[0,0,s,0],[0,0,0,1]])

def rotation_mat(l0,l1):
    # Lines to vectors:
    v0 = l0[1]-l0[0]
    v1 = l1[1]-l1[0]

    #normalize vactors:
    v0_norm = (v0/ np.linalg.norm(v0))
    v1_norm = (v1/ np.linalg.norm(v1))

    # Cross product:
    v_cross = np.cross(v1_norm, v0_norm)

    if np.count_nonzero(v_cross)==0:
        #print("No need to rotate. Vectors are already aligned.")
        return np.identity(4)
    else:
        # c=a⋅b (cosine of angle = dot product)
        c = np.dot(v1_norm,v0_norm)
        # s=‖v‖ (sine of angle)
        s = np.linalg.norm(v_cross)

        # skew-symmetric cross-product matrix of v_cross:
        k = np.array([[0,-v_cross[2],v_cross[1]],[v_cross[2],0,-v_cross[0]],[-v_cross[1],v_cross[0],0]])
        I = np.identity(3)
        # Rotation matrix equation:
        r = I + k + k@k * ((1-c)/(s**2))
        rotat_mat = np.pad(r, ((0,1),(0,1)), 'constant', constant_values=0)
        rotat_mat[3,3] = 1
        return rotat_mat

# First move center line segment to the 0,0,0
def translation_mat(l1):
    xyz = [0,0,0]-l1[0]

    transl_mat = np.identity(4)
    transl_mat[0:3,3] = xyz 
    return transl_mat

# After all the transformation is done but to 0,0,0, move it to the position of the segment
# And giving it a complete shift so it will all be in positive values for 
# scipy affine transformation to work smoothly:
def translation_mat_final_step(l0):
    xyz = l0[0] 

    transl_mat = np.identity(4)
    transl_mat[0:3,3] = xyz 
    return transl_mat


# After all the transformation is done but to 0,0,0, move it to the position of the segment
# And giving it a complete shift so it will all be in positive values for 
# scipy affine transformation to work smoothly:
def translation_mat_final_step(l0):
    xyz = l0[0] 

    transl_mat = np.identity(4)
    transl_mat[0:3,3] = xyz 
    return transl_mat

# Full transformation matrix (available thanks to homogenious coordinates)
# TransformedVector = TranslationMatrix * RotationMatrix * ScaleMatrix * OriginalVector.

def transformation_mat(l0, l1):
    #return np.dot(np.dot(scaling_mat(l0,l1),rotation_mat(l0,l1)),translation_mat(l0,l1))
    return np.dot(translation_mat_final_step(l0),
                  np.dot(np.dot(scaling_mat(l0,l1),rotation_mat(l0,l1)),translation_mat(l1)))


############################# Register ############################

if im_type is 'w':
    worm_file = 'worm_fused'
elif im_type is 'n':
    worm_file = 'nuclei_mask'
elif im_type is 'm':
    worm_file = 'mask'

# define template midline:
template_points = np.asarray([[template_beg[0],template_beg[1],template_beg[2]+i] for i in range(0,num_of_segs*seg_siz_in_template+1,seg_siz_in_template)])

#################### JUST FOR TESTING ######################
# Template line not on one dim:
#template_points = np.asarray([[template_beg[0],template_beg[1]+i,template_beg[2]+i] for i in range(0,num_of_segs*seg_siz_in_template+1,seg_siz_in_template)])

templates_mid_point = [(template_points[i]+template_points[i+1])/2 for i in range(num_of_segs)]
    
# check if the registration to segments already exist and all needed is to combine the segments:
if runonly:
    worm_reged_to_segs = np.asarray([tif.imread(folder + im_type + '_registered_2seg' + str(i) + "_morph" + str(morphstep) + '.tif') for i in range(num_of_segs)])
else:

    worm = tif.imread(folder + worm_file + '.tif')
    print(f'read worm minimum {np.min(worm)}')

    t = time.time()

    # determine segments size:
    seg_siz = int(len(skelet_pix) / num_of_segs)

    print('SEGEMENT SIZE:' + str(seg_siz), flush=True)

    # Pixels in skeleton to poins defining segments (lines) of the center line:
    center_line_points = np.asarray(skelet_pix[0::seg_siz])

    # find the midpoint of each line:
    lines_mid_point = [(center_line_points[i]+center_line_points[i+1])/2 for i in range(num_of_segs)]

    # If we just want the final registration (to the straight template line)
    if morphstep==1:
        # Find transformation martices for all lines:
        transform_mats = [transformation_mat([template_points[i],template_points[i+1]],
                                             [center_line_points[i],center_line_points[i+1]]) for i in range(num_of_segs)]
    # If we want to make a video and need some steps in between:
    else:
        # Find transformation martices for all lines:
        transform_mats = [transformation_mat(
            [(template_points[i]-center_line_points[i])*morphstep+center_line_points[i],
            (template_points[i+1]-center_line_points[i+1])*morphstep+center_line_points[i+1]],
            [center_line_points[i],center_line_points[i+1]]) for i in range(num_of_segs)]

    ##Didn't work
    # If it's run to create the morph video from the original image to the registered one:
    #if morphstep<1:
    #	morph_mat = np.identity(4)
    #	morph_mat[0:3,3] = [morphstep,morphstep,morphstep]
    #	transform_mats = [t_m*morph_mat for t_m in transform_mats]

    # Find the inverse matrix for each transformation matrix:
    inv_transform_mats = [np.linalg.inv(t) for t in transform_mats]

    print('Calculated Transformation Mats + Inv Trans Mats. Time: ' + str(time.time()-t), flush=True) 
    t = time.time()

    # Clean worm file outside of mask (=0)
    worm[worm_mask==False] = 0

    # Apply transformation of each segment on the worm:
    worm_reged_to_segs = [affine_transform(worm.astype('int32'), itm, output_shape=reged_siz, order=interpulation_order) for itm in inv_transform_mats]
    #### for debuging:
    min_in_worm_registered_to_segs = np.min(np.asarray([np.min(w) for w in worm_reged_to_segs]))
    print(f'min worm registered to segs {min_in_worm_registered_to_segs}')

    print('Calculated worm image registered to each segment. Time: ' + str(time.time()-t), flush=True) 

    # Saving each image - of worm aligned to each segment:
    [tif.imsave(folder + im_type + '_registered_2seg' + str(i) + '_morph' + str(morphstep) + '.tif', w) for i,w in enumerate(worm_reged_to_segs)]

t = time.time()

#worm_reged_to_segs = [tif.imread(folder + 'registered_2seg' + str(i) + '.tif') for i in range(num_of_segs)]

worm_reged_to_segs = np.asarray(worm_reged_to_segs, dtype=np.int32)
print (f'worm registered to segs {np.min(worm_reged_to_segs)}')

# Find the distance between each point in the image the midpoint of each segment of the center line:
# Because its a long calculation, do it only by z value (as center line values differ only in z):
# (For each z value we determine the distance of the entire 2d image to have the same value)
# (Might not be accurate, but saves computation time)

registered_worm = np.zeros(reged_siz, dtype=np.int32)
for z in range(worm_reged_to_segs[0].shape[2]):
    dist_to_segs = [abs(z-t[2]) for t in templates_mid_point]
    inv_dist = [1/d**alpha if d>0 else 1 for d in dist_to_segs]
    multiplier_4_nor = 1/sum(inv_dist)
    aff_2_segs = [d*multiplier_4_nor for d in inv_dist]
    
    registered_worm[:,:,z] = np.sum(np.multiply(worm_reged_to_segs[:,:,:,z], np.asarray(aff_2_segs)[:, np.newaxis, np.newaxis]), axis=0)
    print(f'registered worm min {np.min(registered_worm)}')

print('Calculated final registration. Time: ' + str(time.time()-t), flush=True) 

tif.imsave(folder + 'registered_worm' + wormnum + im_type + '_a' + str(alpha) + '_interpulation_order' + str(interpulation_order) + 
    '_nsegs' + str(num_of_segs) + '_sseg' + str(seg_siz_in_template) + '_morph' + str(morphstep) + '.tif', registered_worm.astype(np.int16))


