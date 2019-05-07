from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import os
import random
import math
import pickle
import copy
from skimage.io import imread
from typing import Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.morphology import (label,
                                erosion,
                                dilation,
                                remove_small_objects,
                                remove_small_holes,
                                square,
                                opening)      
from skimage.filters import gaussian
from skimage.exposure import (rescale_intensity,
                              adjust_sigmoid,
                              )
from skimage.feature import (structure_tensor, 
                             structure_tensor_eigvals,
                             )
from skimage.transform import (PiecewiseAffineTransform,
                               warp,
                               rescale,
                               )

"""A module for preprocessing image mask data for various nuclei mask data.
"""


class image_reader(object):
    """Image reader and mask maker from path/id_. 
   
    Imports image data and creates and integer labeled masks from mask data. 
    Attributes:
        path: directory path to images data
        id_: str input id of file that has images and mask subdirectories. 
    """
    def __init__(self, path, id_):
        """Initialization."""
        self.image = img = imread(os.path.join(path,id_, "images", id_+".png"))
        self.id = id_
        self.mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
        index = 1
        
        for mask_file in next(os.walk(os.path.join(path,id_ ,'masks')))[2]:
            mask_= imread(os.path.join(path, id_, "masks", mask_file))
            mask_[mask_>0] = 1
            self.mask += index*mask_
            index += 1
    @property
    def image_mask(self):
        """Property for returning image and integer labeled masks."""
        return self.image, self.mask
    
    
def cutter(data,
           size, 
           mask_tolerance = False,
           image_channels = 1,
           ):
    """Cuts image and or mask data up into square pieces.
    
    An image is cut into squares and returned as a batch. In an attempt to use 
    all the data possible, it resizes some images if their size is almost 'size'.
    Also, it takes 'edge scraps' and includes cuts along those edges (there will
    be some overlap in the image partitions when this happens).  
    If input is a  topological mask type then it should be on data[:,:,2:5].
    If integer mask type then it should be on data[:,:,2].
    Args:
        data: image data 
        size: size of squares to cut height/width features 
        mask_tolerance: int area where if a square image has a mask with an area
            less than this, the cut isn't used
        image_channels: number of channels for the image data
    Returns:
        ndarray of batched image cuts from data
    Raises: 
        ValueError: if the data is too small to be even resized before being cut
    """
    
    dtype = data.dtype
            
    if len(data.shape) <= 2:
        data = data[...,np.newaxis]
        
    s = int(size)
    y, x, ch = data.shape
    first = image_channels
    second = ch
    mask_type = second - first
    
    if y <= s*1.1 or x <= s*1.1:
        m = min(x, y)
        if second - first >= 1:
            img = rescale(data[..., 0:image_channels], 
                          s/m,
                          preserve_range = True,
                          )
            mask=rescale(data[..., first:second],
                         s/m,
                         order = 0,
                         preserve_range = True,
                         )
            data = np.concatenate([img, mask], axis = -1).astype(dtype)
        else:
            data = rescale(data, s/m, preserve_range = True).astype(dtype)
            
    if mask_type > 1:
        first += 1  
        
    if mask_type > 0 and mask_tolerance == False:
        mask_tolerance = 0

    ry, rx = y%s, x%s
    ny, nx=int((y - ry) / s), int((x - rx) / s)
  
    if ny == 0 or nx == 0: 
        raise ValueError("The image was too small in one of the dimensions.")
    
    def append(dat, old, ii, jj):        
        new = dat[s*jj:s*(jj+1), s*ii:s*(ii+1), :].reshape(1, s, s, -1)
        if mask_type>0:
            if np.sum(new[..., first:second]>0) >= mask_tolerance:
                if old.all() != None:
                    new = np.concatenate((old, new),axis = 0)
                return new
            else:
                return old
        else:
            if old.all() != None:
                new = np.concatenate((old, new), axis = 0)
            return new
        
    arr = np.array(None)
    for i in range(nx):
        for j in range(ny):
            arr = append(data, arr, i, j)
            
    
    if ry >= 50:
        data_flip = np.flip(data, 0)
        for j in range(ny):
            arr = append(data_flip, arr, 0, j)
    if rx >= 50:
        data_flip=np.flip(data, 1)
        for i in range(nx):
            arr = append(data_flip, arr, i, 0)
    if arr.all() == None:
        arr=data[0:s, 0:s, :].reshape(1, s, s, -1)
    return arr  

def fill_int_holes(arr, 
                   area = 25,
                   smoothing = 1,
                   remove_objects : Optional[int] = False,
                   ):
    """Fills holes, smoothes, and cleans up in an image/mask.
    
    For either postprocessing or preprocessing, some images have missing pixels
    on their interior masks. This fills them. It also has an option to smooth 
    boundaries as well as remove objects. 
    Note that in some images a viewer might question whether to fill a region 
    with one integer value over another. This is resolved by simply going with 
    the larger number. 
    Args: 
        arr: image mask
        area: maximal area of interior region that will be filled
        smoothing: number of iteration to run smoothing of boundary with 
            skimage.morphology.opening
        remove_objects: optional int to be the minimial area of an object kept
            in the image
    Returns: 
        Image data array of same shape as arr.
        """
    
    arr = arr.astype(np.int32)
    if type(remove_objects) == int:
        arr = remove_small_objects(arr, 
                                   min_size = remove_objects,
                                   connectivity = 2,
                                   )
    r = remove_small_holes(arr,area,connectivity=2)
    z=(arr==0)
    holes=r*z
    marked_holes=arr-holes
    while np.any(marked_holes<0):
        marked_holes=np.where(holes,dilation(marked_holes,selem=square(3)),marked_holes)
        holes=(marked_holes<0)
    new_arr=marked_holes
    for i in range(smoothing):
        new_arr=opening(new_arr,selem=square(3))
    
    return new_arr


def inspect_round_1(good_mask_collection,
                    bad_mask_collection,
                    uninspected_collection,
                    ):
    """First round of preprocessing interactively. 
    
    Sort images into ones with good masks and bad masks (or ones in need of 
    repair). There are options to do small fixes (fill in holes and smooth 
    edges). This is made specifically for 9 collections of image types in the 
    dataset. 
    This allows a user to run through images to compare with their masks to find
    any odd data. 
    Args:
        good_mask_collection: dict with keys 0-8, the collection number, with 
            the already processed images
        bad_mask_collection: dict with keys 0-8, the collection number, with 
            the images with severe mask errors.
        uninpected_collection: dict with keys 0-8, the collection number, with 
            uninspected images
    Returns:
        Three updated dictionaries in args, in that order, as well as a marker 
        str for quitting or continuing the inspections.
    """
    
    good = good_mask_collection
    bad = bad_mask_collection
    coll = uninspected_collection
    keys = sorted(list(coll.keys()))
    while True:
        numb = input("Which collection do you want to look at (0-8):")
        try:
            collection_number = int(numb)
            if collection_number >= 0 and collection_number <= 8:
                break
            else:
                print("Out of range")
                continue
        except Exception as err:
            print(f"An error occured: {err}")
            continue
        
    collection_key = keys[collection_number]
   
    while True:
        if 0 < len(coll[collection_key]):
            original_image_class = coll[collection_key][0]
            image_class = copy.deepcopy(original_image_class)
        else:
            print("We're done with that collection")
            cont = ""
            break
        
        image, mk= image_class.image_mask
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 12))
        ax[0].imshow(image)
        ax[1].imshow(mk, plt.get_cmap("nipy_spectral_r"))
        plt.show()
        
        def fixit(mask):
            area, smoothing = 4, 0
            while True:
                new_mask = fill_int_holes(mask, 
                                          area = area,
                                          smoothing = smoothing,
                                          )
                fig, ax=plt.subplots(2, 2, figsize = (10, 8), squeeze = True)
                ax[0, 0].imshow(mask, plt.get_cmap("nipy_spectral_r"))
                ax[0, 0].set_title("Original Mask")
                ax[0, 1].imshow(new_mask, plt.get_cmap("nipy_spectral_r"))
                ax[0, 1].set_title("New Mask")
                ax[1, 0].imshow(image)
                ax[1, 1].imshow(image)
                plt.show()
                
                if (new_mask != mask).any():
                    print(f"There were holes filled of size <= {area}.")
                else:
                    print(f"There weren't any holes filled of size <= {area}.")
                    
                cont=""
                
                while cont != "y" and cont != "n" and cont != "a":
                    cont = input("Replace old mask with new (y or n) or adjust parameters (a)?:")
                               
                if cont == "y":
                    image_class.mask=new_mask
                    break
                if cont == "n":
                    break
                if cont == "a":
                    keep = ""
                    while keep != "y" and keep != "n":
                        keep = input("Work the adjusted mask (y) or work the original mask(n):")
                    if keep == "y":
                        mask = new_mask
                    area = int(input("Area:"))
                    smoothing = int(input("Smoothing iteration number:"))
                    continue
                        
                
                    
        def inspect():
            """Inspect the cut images."""
            cut_image = cutter(image, 0, 64)
            cut_mask = cutter(image_class.mask, 0, 64)
            print(cut_mask.shape, cut_image.shape)
            l = cut_image.shape[0]
            count = 0
            while count < l:
                fig, ax = plt.subplots(1, 2, figsize = (10, 12))
                ax[0].imshow(cut_image[count, ...])
                ax[1].imshow(cut_mask[count, ...][..., 0],
                             plt.get_cmap("nipy_spectral_r"),
                             )
                plt.show() 
                cont = input("Press ENTER to continue or 'a' to abort:")
                if cont == "":
                    count += 1
                    continue
                else:
                    break
                
        while True:           
            fix = input("Do you wish to fill and smooth mask (y for yes or ENTER for no)?")
            if fix == "y":
                fixit(image_class.mask)
            cont = input("Do you wish to inspect further or continue? (y for yes ENTER for no):")
            if cont == "y": 
                inspect()
            again = input("Shall we inspect and/or fill again?(y for yes ENTER for no)")
            if again == "y":
                continue
            else:
                break
                
        opinion = "nope"
        while opinion != "g" and opinion != "b" and opinion != "":
            opinion = input("Press 'g' if a good image, 'b' for bad, 'ENTER' for deal later:")
        
        if opinion == "g":
            good[collection_key].append(image_class)
            coll[collection_key].remove(original_image_class)
        if opinion == "b":
            bad[collection_key].append(image_class)
            coll[collection_key].remove(original_image_class)
            
        cont = input("Press ENTER to continue or 'q' to quit:")
        if cont == "":
            continue
        else:

            break
    return good, bad, coll, cont



def first_inspection(good, bad, coll):
    """Runs a loop for inspect_round_1.
    Args:
        good: good image dict as in inspect_round_1
        bad: bad image dict as in inspect_round_1
        coll: uninspected image dict as in inspect_round_1
    Returns: 
        Dictionary of good, bad, and uninspected images
    """
    cont = ""
    while cont != "q":
        good, bad ,coll, cont = inspect_round_1(good, bad, coll)
        if sum([len(x) for x in coll.values()]) == 0:
            print("YOU'RE DONE")
            break

    return {"good":good,"bad":bad,"uninspected":coll}

def run_inspection(path):
    """Runs the round 1 inspection. 
    
    Pulls a pickled dict of image data for inspection. The dict has keys "good",
    "bad", and "uninspected". It also has options to save the data after 
    inspection. 
    Args:
        path: str path to directory with saved image dictionary
    """
    while True:
        comm = input("Would you like to use the stored good/bad/uninspected images? (y or n):")
        if comm != "y" and comm != "n":
            continue
        else:
            break
    
    
    if comm == "y":
        with open(path + "/inpected_images.pkl", "rb") as f:
            old_inspected = pickle.load(f)
    
        good = old_inspected["good"]
        bad = old_inspected["bad"]
        coll = old_inspected["uninspected"]
        
    if comm == "n":
        
        TRAIN_PATH = path + "stage1_train/"
        train_ids = next(os.walk(TRAIN_PATH))[1]
        print("Images Loading...")
        image_collection = [image_reader(TRAIN_PATH, img) for img in tqdm(train_ids)]
        print("Loaded Images")
    
    
        shape_dict = {}
        for x in tqdm(image_collection):
            m = x.image_mask[0].shape
            if m in shape_dict.keys():
                shape_dict[m].append(x)
            else:
                shape_dict[m] = [x]
                
        good = {k: [] for k in shape_dict.keys()}
        bad = {k: [] for k in shape_dict.keys()}
        coll = shape_dict
        
    inspected = first_inspection(good, bad, coll)
    
    g = sum([len(x) for x in inspected["good"].values()])
    b = sum([len(x) for x in inspected["bad"].values()])
    u = sum([len(x) for x in inspected["uninspected"].values()])
    print(f"Number of good images: {g}.\n Number of bad: {b}.\n Number of uninspected: {u}.")
    
    comm = ""
    while comm != "y" and comm != "n":
        comm = input("Would you like to store the good/bad/uninspected images? (y or n):" )
    if comm == "y":
        with open(path+"/inpected_images.pkl", "wb") as f:
            pickle.dump(inspected, f)

def local_f(arr, f, w: int = 3):
    """Implements f locally on all windows of arr of size w by w. 

    Args:
        arr: a 2D array,
        f: a function acting on flattened array of size w**2.
        w = windowsize.
    Returns:
        numpy array of values from f acting on w by w windows from arr
    """
    p = int((w - 1) / 2) #pad_size
    arr_pad = np.pad(arr, p, mode="constant")
    it = np.nditer([arr, None], order = "C", flags = ['multi_index'])
    for x, y in it:
        a, b = it.multi_index
        y[...] = f(arr_pad[a: a + w, b: b + w].flatten())
    return it.operands[1]

def boundary_knn(arr):
    """Turns Boundary, Interior, Exterior mask into single integer labeled mask
    
    This takes interior regions, labels nontouching ones with integers, and 
    appends their respective boundaries by giving boundary pixels a value 
    corresponding to a majority vote on its nearset neighbors (defined by '2-
    connectivity'). Thus the def name's reference to 'k nearest neighbors'.
    This is a near inverse of boundary_mask_full.
    
    It will fail on regions whose boundary becomes 1 dimensional with 
    disconnected interiors and/or long portions of 1D boundary. It is a more 
    vectorized knn for k=1 with respect to the max/sup norm on pixel coordinates 
    specialized to focus most computation time on boundary.
    
    Args:
        arr: numpy arry arr.shape=(N,M,3) which is a mask for background, 
            interior, and boundary.
    Returns: 
        numpy array of shape (N,M) with integer labeled mask regions. 
    """
    arr = arr.astype(np.int32)
    N, M, _ = arr.shape
    labeled_boundaries = -1 * arr[:, :, 2]
    regions = label(arr[:, :, 1], connectivity = 2) 
    def mode_for_bdy(x):
        if x[4] != -1:
            return 0
        values, counts = np.unique(x, return_counts = True)
        if values.max() > 0:
             return values[values > 0][counts[values > 0].argmax()].astype(np.int32)
        else:
            return -1
    i = 0
    while True:
        r = regions + labeled_boundaries
        labeled_boundaries = local_f(r, mode_for_bdy, 3)
        regions += np.where(labeled_boundaries>=1, labeled_boundaries, 0)
        i += 1
        if np.any(labeled_boundaries == -1) and i < 2:
            labeled_boundaries=np.where(labeled_boundaries == -1, -1, 0)
        else:
            break
    return regions

def boundary_knn_batch(arr):
    """Enacts boundary_knn on a batch.
    Args:
        arr: batch array of data to be fed into boundary_knn
    Returns:
        batched array of integer labeled masks
    """
    l = [boundary_knn(arr[i, ...])[np.newaxis, ...] for i in range(arr.shape[0])]
    return np.concatenate(l,axis = 0)
        
def boundary_mask_full(mask_in):
    """Turns integer labeled mask to background, interior, boundary mask.
    
    This is a near inverse of boundary_knn. For last index slice: 0:background
    mask, 1:interior mask, 2:boundary.
    Args:
        mask_in: (N,M,channels) array of integer labeled masks
    Returns: 
        (N,M,3) array of background, interior, boundary mask.
    """
    rank = len(mask_in.shape)
    if rank == 3:
        mask_in = mask_in[0, ...]
    labels = int(np.max(mask_in))

    x, y = mask_in.shape[0], mask_in.shape[1]
    labeled_topo = np.zeros((x, y, 3), dtype = np.float32)
    labeled_topo[:, :, 0] = np.where(mask_in == 0, 1, 0)
    for l in range(labels):
        nuc = (mask_in == l+1)
        m = erosion(nuc, selem = square(3))
        labeled_topo[:, :, 1] += m
        labeled_topo[:, :, 2] += nuc^m
    if rank == 3:
        labeled_topo = labeled_topo[np.newaxis, ...]  
    return labeled_topo

def grey(image):
    """Turn 3 channel image to greyscale.
    Args:
        image: 3 channel image data
    Returns: 
        average of 3 channels image data"""
    if np.max(image) >= 256:
        print("Image range error. Not in uint8 range.")
        return
    img = image[..., 0: 3].astype(np.float32) / 255
    img = np.mean(img, axis = -1) 
    return img

def normalize_image(img):
    """Normalize images.
    
    Normalize images by gaussian smoothing for noise, push background values to 
    0, and nuclei values toward 1 with sigmoid. The set parameters seemed not to 
    damage boudnary information between nuclei too much. 
    Args: 
        img: greyscale image data
    Returns: 
        normalized image
    """
    img_2 = gaussian(img, sigma = 1)
    cut = np.mean(img_2)
    sig = adjust_sigmoid(img_2, 
                         cutoff = cut,
                         gain = 5,
                         inv = (cut > 0.5),
                         )
    sig = rescale_intensity(sig)
    
    sig_2 = gaussian(sig, sigma = 1)
    A = structure_tensor(sig_2, sigma = 0.8, mode = "reflect")
    e1, e2 = structure_tensor_eigvals(*A)
    d = e1 + e2
    str_corr = np.where(d > np.finfo(np.float16).eps, 
                        (e1 - e2) ** 2,
                        0,
                        )
    bge = str_corr > np.mean(str_corr) * .01
    
    bdy_grad_eigen = opening(remove_small_objects(bge, 
                                                  min_size = 64, 
                                                  connectivity = 2,
                                                  ),
                             ) 
    gaus = gaussian(sig, sigma =1 )
    out = np.where(bdy_grad_eigen, sig, gaus)
    out = rescale_intensity(out)
    return out.astype(np.float32)
        
def grey_bend(a,b,c,im):
    """Transform 3 channel data to greyscale with weights.
    Args:
        a: first channel weight
        b: second channel weight
        c: third channel weight
        im: 3 channel image data
    Returns:
        greyscale image
    """
    return im[..., 0] * a + im[..., 1] * b + im[..., 2] * c

def stru(im):
    """Attempts to find edges in the image through use of structure tensor.
    
    Computes a mask of values that are higher mostly on edges by using the 
    square difference of the structure tensor eigen values. 
    Args: 
        im: input greyscale image
    Returns:
        mask of values higher on edges than interiors (see above)
    """
    A = structure_tensor(im, sigma = 0.8, mode="reflect")
    e1, e2 = structure_tensor_eigvals(*A)
    d = e1 + e2
    st = np.where(d > np.finfo(np.float16).eps, (e1 - e2) ** 2, 0)
    return st[..., np.newaxis]

def bias_mean(arr):
    """Computes a biased mean of the 3 channels for an image. 
    Args:
        arr: 3 channel image
    Returns:
        biased mean of 3 channel image
    """
    a = np.mean(arr[..., 0])
    b = np.mean(arr[..., 1])
    c = np.mean(arr[..., 2])
    if a + b + c == 0:
        return 0
    return (a ** 2 + b ** 2 + c ** 2) / (a + b + c)

def rm(x):
    """Composition of opening and remove small objects to decrease noise.
    Args: 
        x: image 
    Returns:
        image 
        
    """
    return opening(remove_small_objects(x, min_size = 64, connectivity = 2))

def normalize_image_3chan(img):
    """Normalize an image without damaging boundaries too much. 
    
    Gaussian smooth everything not an edge (using structure tensor), make 
    background black = 0, push values toward 0 or 1 with sigmoid while being 
    careful not to damage information between adjacent nuclei too much.
    Args:
        img: 3 channel image
    Returns:
        normalized greyscale image
    """

    if len(img.shape) == 2:
            img = np.repeat(img[..., np.newaxis], 3, axis = -1)
    elif len(img.shape) < 2 or len(img.shape) > 4:
        raise ValueError("This is not an image. The dimensions are {}.".format(img.shape))
        
    img = img[..., 0: 3]
    dtype = img.dtype
    
    if dtype != np.float:
        img = img.astype(np.float)
        if dtype == np.uint8:
            img = img / (2 ** 8 - 1)
        elif dtype == np.uint16:
            img = img /(2 ** 16 - 1)
        elif dtype == np.uint32:
            img = img /(2 ** 32 - 1)
        elif dtype == np.uint64:
            img = img /(2 ** 64 - 1)
        else:
            raise ValueError("This image has unknown data type.")
    else:
        if np.max(img) > 1:
            img = img / np.max(img)
    
    if np.all(np.max(img, axis = -1) == np.min(img, axis = -1)):
        img = img[..., 0]
        blur = gaussian(img, sigma = 1)
        blur1 = gaussian(img, sigma = 2)
        cut = np.mean(blur1)
        blur2 = adjust_sigmoid(blur1, cutoff = cut, gain  =8)
        str_corr = stru(blur1)[..., 0]
        str_corr2 = stru(blur2)[..., 0]
        bge = str_corr > np.mean(str_corr) * 0.01
        bge2 = str_corr2 > np.mean(str_corr2) * 0.01
        both = np.maximum(rm(bge), rm(bge2))
        out = np.where(both, img, blur1)
    else:
        blur1 = gaussian(img, sigma = 2, multichannel = True)
        cut = np.mean(blur1)
        blur2 = adjust_sigmoid(blur1, cutoff=cut, gain = 8)
        
        str_corr = np.concatenate((stru(blur1[..., 0]),
                                   stru(blur1[..., 1]),
                                   stru(blur1[...,2])),
                                  axis=-1,
                                  ) 
        str_corr2=np.concatenate((stru(blur2[...,0]),
                                  stru(blur2[...,1]),
                                  stru(blur2[...,2]),
                                  ),
                                 axis = -1,
                                 ) 
        bge = str_corr > bias_mean(str_corr) * 0.1
        bge2 = str_corr2 > bias_mean(str_corr2) * 0.1
        
        mask1 = np.concatenate((rm(bge[...,0:1]),
                                rm(bge[...,1:2]),
                                rm(bge[...,2:3]),
                                ),
                               axis = -1,
                               )
        mask2 = np.concatenate((rm(bge2[...,0:1]),
                                rm(bge2[...,1:2]),
                                rm(bge2[...,2:3]),
                                ),
                              axis = -1,
                              )
        both = np.maximum(mask1, mask2)
        sd = np.std(str_corr, axis = (0, 1))
        sd = sd / np.sum(sd)
        blur = gaussian(img, sigma = 1 - (sd / 2), multichannel = True)
        out = np.where(both, img, blur)
        out = grey_bend(*sd, blur)
        both=np.max(both, axis = -1)
   
    if np.mean(out) > 0.5: #If perhaps the image has 1: background
        out = 1 - out
        
    out = rescale_intensity(out)
    return out.astype(np.float32)


class random_distort(object):
    """Random distortions of images for expanding training set. 
    
    Attributes:
        image: 3 channel image dtype = unit8 plus 4th channel all 1's for 
            boolean mask to crop image after distortion. The methods below put
            0's in channel where there no image after distorting edges in inward
            motions. 
        mask: integer labeled masks in single channel 
        
    """  
    
    def __init__(self,image,mask):
        """Initialization of variables and creates warped image and mask.
        
        Utilizing skimage.transform.PiecewiseAffineTransform, a mesh grid is 
        made and distorted. The parameters chose below seemed to not distort 
        too much so as to keep nuclei recognizable. 
        """
        
        self.image = image
        self.mask = mask
        
        def cropper(img, mask):
            """Crops img and mask after distortion to clean up edges. 
            
            Args: 
                img: see self.image
                mask: see self.mask
            Returns:
                img: cropped img
                mask: cropped mask
            """
            ss_r = int(sd_rows) + 1
            ss_c = int(sd_cols) + 1
            for k in range(10):
                if np.all(img[:, :, 3] > 0):
                    break
                img = img[ss_r: -ss_r, ss_c: -ss_c, ...]
            k_c = ss_c * k
            k_r = ss_r * k
            if k > 0:
                mask = mask[k_r: -k_r, k_c: -k_c]
            return img, mask
    
        rows, cols = image.shape[0], image.shape[1]
        num_rows = math.ceil(rows / 30) + 1
        num_cols = math.ceil(cols / 30) + 1
        sd_rows = rows / (8 * (num_rows - 1))
        sd_cols = cols / (8 * (num_rows - 1))
        src_cols = np.linspace(0, cols, num_cols)
        src_rows = np.linspace(0, rows, num_rows)
        src_rows, src_cols = np.meshgrid(src_rows, src_cols)
        src = np.dstack([src_cols.flat, src_rows.flat])[0]
        distort = [sd_cols, sd_rows] * np.random.uniform(-2, 2, src.shape)
        dst_rows = src[:, 1] - distort[:, 1]
        dst_cols = src[:, 0] - distort[:, 0]
        dst = np.vstack([dst_cols, dst_rows]).T
        tform = PiecewiseAffineTransform()
        tform.estimate(src, dst)
        
        self.warp_im = im = warp(self.image,tform, 
                                 output_shape=(rows, cols),
                                 preserve_range = True,
                                ).astype(np.uint8)
        
        self.warp_mk = mk = warp(self.mask,
                                 tform,
                                 output_shape=(rows, cols),
                                 preserve_range= True,
                                 order = 0,
                                 ).astype(np.int32)
        self.crop_image, self.crop_mask = cropper(im, mk)
        self.crop_mask = opening(self.crop_mask, selem=square(3))
       
    def view_images(self):
        """Inspect pre and post cropped images.
        """
        fig, ax = plt.subplots(2,2,figsize=(10,10))
        ax[0,1].imshow(self.warp_mk)
        ax[0,0].imshow(self.warp_im)
        ax[1,0].imshow(self.crop_image)
        ax[1,1].imshow(self.crop_mask)
        plt.show()
        
    @property
    def distort(self):
        """Property cropped and distored image and mask."""
        return self.crop_image, self.crop_mask

    @property
    def distort_image(self):
        """Property cropped and distorted image"""
        return self.crop_image
    @property
    def distort_mask(self):
        """Property cropped and distorted mask"""
        return self.crop_mask
        

def distort_cut_transform(img_read_list,
                          distort_number,
                          normalize_function
                          ):
    """The main function used to expand image/mask collection. 
    
    Processes image by creating several sets of distorted images, randomly flips,
    crops the images to 240x240 pieces, concatenates into a batch, 
    normalizes the images with the normalize function, and creates a boundary,
    interior, background 3 channel masks.
    Args:
        img_read_list: either image_read class object or list of pairs of 
            image and masks
        distort_number: Number of rounds of distorting images. Or just the 
            approximate multiplier for expanding the data set (cropping will 
            make the actual multiplier less)
        normalize_function: function for normalizing images. Inputs 3 channel
            image and outputs 1 channel. e.g. normalize_image_3chan 
            
    Returns:
        Batched image/mask of shape [batch,N,M,channels] where the slice 
        channels = 0 is the image and channels = [1:4] is the boundary, interior,
        background masks. 
    """
    try:
        imgs = [x.image_mask for x in img_read_list]
    except:
        imgs = img_read_list
        
    precut = imgs[:]
    
    def flip(img, msk):
        """Randomly flip image and mask.
        Args:
            img: image 
            msk: mask
        """
        x, y = random.randint(0, 1), random.randint(0, 1)
        if y==1:
            img=np.flip(img,0)
            msk=np.flip(msk,0)
        if x==1:
            img=np.flip(img,1)
            msk=np.flip(msk,1)
        return img,msk
    
    for i in range(distort_number):
        print("Distorting round {}.".format(i+1))
        temp = [flip(*random_distort(*img_msk).distort) for img_msk in tqdm(imgs)]
        precut.extend(temp)
        
    print("Finished all distorting.")
    
    precut = [np.concatenate([img,msk[..., np.newaxis]], axis = -1) \
              for (img, msk) in precut]
    
    print("Now cutting....")
    
    postcut_temp = [cutter(img, 240, mask_tolerance = 64, image_channels = 4) \
                    for img in tqdm(precut)]
    
    L = len(postcut_temp)
    M = math.ceil(L / 50)
    print("Concatenate round 1...")
    temp = [np.concatenate(postcut_temp[i * 50: (i + 1) * 50], axis = 0) \
            for i in tqdm(range(M))]
    print("Finishing concatenating...")
    postcut = temp[0]
    for x in tqdm(temp[1:]):
        postcut = np.concatenate([postcut, x], axis = 0)
    print(postcut.shape)
    length = postcut.shape[0]
    postransform = np.zeros((length, 240, 240, 4),dtype=np.float32)
    
    print("Now transforming to greyscale, normalizing, and creating topo_mask...")
    for i in tqdm(range(length)):
        postransform[i, ..., 0] = normalize_function(postcut[i,...,0:3].astype(np.float) / 255)
        postransform[i, ..., 1: 4] = boundary_mask_full(postcut[i, ..., 4])
    return postransform
        
def process_save_images(STORE_DIR,image_mask,distortion_count,plot=False):
    """
    Args:
        STORE_DIR: directory where the processed images will be saved
        image_mask: mage_mask is either a list of image_reader objects or a list 
            of tuples of (image4channels, integer_mask)
        distortion_count:
        plot: boolean 
    Returns:
        
    """
    from datetime import date
    
    day = str(date.today())
    print("Distort cut transform....")
    arr = distort_cut_transform(image_mask, distortion_count) 
    print("Saving to {}.".format(os.path.join(STORE_DIR, "training_ready_{}".format(day))))
    np.savez_compressed(os.path.join(STORE_DIR, "training_ready_{}".format(day)), arr)
    print("Finished saving.")
    if plot:
        fig,ax=plt.subplots(2,2,figsize=(10,10))
        ax[0,0].imshow(arr[0,...,0],cmap="gray")
        ax[0,1].imshow(arr[0,...,2],cmap="gray")
        ax[0,1].set_title("Interior")
        ax[1,0].imshow(arr[0,...,3],cmap="gray")
        ax[1,0].set_title("Boundary")
        ax[1,1].imshow(arr[0,...,1],cmap="gray")
        ax[1,1].set_title("Background")
        plt.show()
    if np.all(arr[0,...,1:4] == 0): 
        print("Something is wrong")

#
def image_topo_mask_viewer(arr):
    """Inspecter for postprocessed images and masks 
    
    Allows the viewer to look through a batch of images and masks individually.
    Args:
        arr: shape = (batchsize,y_dim,x_dim, 4) 1st channel = image and last 
            3 channels = topo_mask    
    """
    offset = int(input("What image offset:"))
    for i in range(offset,arr.shape[0]):
        fig, ax = plt.subplots(2, 2, figsize = (10, 10))
        ax[0,0].imshow(arr[i, ..., 0], cmap = plt.get_cmap("gray"))
        cmap = plt.get_cmap("rainbow")
        cmap.set_under("blue")
        cmap.set_over("green")
        ax[0, 1].imshow(arr[i, ..., 2],
                        cmap = cmap,
                        vmin = 0.1,
                        vmax = 0.9,
                        )
        ax[0,1].set_title("Interior")
        ax[1,0].imshow(arr[i,...,3],
                       cmap = cmap, 
                       vmin = 0.1,
                       vmax = 0.9,
                       )
        ax[1,0].set_title("Boundary")
        ax[1,1].imshow(arr[i, ..., 1],
                       cmap = cmap, 
                       vmin = 0.1, 
                       vmax = 0.9,
                       )
        ax[1,1].set_title("Background")
        plt.show()
        
        cont=input("Press ENTER to continue, and any other key to quit:")
        if cont != "":
            break

def eval_test_images_and_run_length_encoding(unet_ckpt_path = None,
                                             test_imgs_path = None,
                                             csv_file_path = None,
                                             model_layers = None,
                                             model_activation = None,
                                             normalize_function = None,
                                             remove_smalls : Optional[int] = None,
                                             ):
    """Predict images and create csv of run length encoding for predictions.
    
    Takes raw images and preprocesses them, predicts mask, cleans masks,
    preforms run length encoding, and stores the output as a csv. 
    Args:
        unet_ckpt_path: tensorflow model checkpoint path
        test_imgs_path: input image directory path
        csv_file_path: filepath for output csv file
        model_layers: list of number of cells in unet's layers
        model_activation: activation function for cells used in unet model
        normalize_function: normalization function used on training images
        remove_smalls: remove 'peppered" noise in predicted masks of int size.  
    """
    
    if unet_ckpt_path == None or test_imgs_path == None or csv_file_path == None:
        raise ValueError("Need to enter unet_ckpt_path, test_imgs_path, and csv_file_path.")

    from DSB2018_Unet import hyperparams, nuclei_u_net
    import pandas as pd
    import pickle as pkl
    
    params = hyperparams(checkpoint_path = unet_ckpt_path,
                         layers = model_layers,
                         activation = model_activation
                         )
    print("Loading images...")
    image_list = []
    for id_ in tqdm(next(os.walk(test_imgs_path))[1]):
        image_list.append((id_,
                           imread(os.path.join(test_imgs_path,id_,"images",id_+".png")),
                           ),
                          )
    
    print("Preprocessing Images...")
    image_list = [(id_, normalize_function(img)[np.newaxis,...,np.newaxis]) \
                   for id_,img in tqdm(image_list)]
    with open("preprocessed_images.pkl","wb") as f:
        pkl.dump(image_list,f)
    
    shapes_dict = {}
    for elem in image_list:
        key = elem[1].shape
        if key in shapes_dict.keys():
            shapes_dict[key].append(elem)
        else:
            shapes_dict[key]=[elem]
    
    for key in shapes_dict.keys():
        shapes_dict[key] = list(zip(*shapes_dict[key]))
        shapes_dict[key] = (shapes_dict[key][0],
                            np.concatenate(shapes_dict[key][1], axis = 0),
                            )
    
    f = lambda x: {"predict_images":x}
    
    for key in shapes_dict.keys():
        print("Determining topo masks with shape {}...".format(shapes_dict[key][1].shape))
        size = shapes_dict[key][1].shape[1] * shapes_dict[key][1].shape[2]
        params.predict_batch_size = math.ceil(1800000 / size)
        output = [msk["labels"] for msk in nuclei_u_net(None, f(shapes_dict[key][1]), params)]
        shapes_dict[key] = list(zip(shapes_dict[key][0], output))
    
    image_list=[]
    for value in shapes_dict.values():
        image_list.extend(value)
     
    with open("predict_images_temp.pkl","wb") as f:
        pkl.dump(image_list, f)
    print("test dimensions",image_list[0][1].shape)
    print("Converting topo masks to integer enumerated masks...")
    
    image_list = [(id_,boundary_knn(msk)) for id_, msk in tqdm(image_list)]
    print("test dimensions again",image_list[0][1].shape)
    DF = pd.DataFrame(columns = ("ImageId", "EncodedPixels"))
    
    def run_length_encoder(msk):
        """Run length encoder function 
        Args:
            msk: 2D mask of one object
        Returns:
            run length encoded string
        Raises: 
            ValueError if the mask is empty
        """
        y = msk.shape[0]
        l = sorted([(y * i) + j + 1 for j, i in np.argwhere(msk)])
        if l == []:
            raise ValueError("Something is wrong")
        else:
            encoded = [l[0], 1]
            last = l[0]
            
        for n in l[1:]:
            if n == last + 1:
                encoded[-1] += 1
                last += 1
            else:
                encoded.extend([n, 1])
                last = n
                
        return " ".join([str(i) for i in encoded])
    
    def run_length_writer(df, id_, msk):
        """Removes small objects, runs run_length_encoder, puts in df under id_
        Args:
            df: dataframe with run length encoded strings. cols = "ImageId", 
                "EncodedPixels"
            id_: image id string 
            msk: integer labeled masks for an image
        Returns:
        """
        M = np.max(msk)
        for i in range(1, M + 1):
            if remove_smalls != None:
                if np.sum(msk == i) <= remove_smalls:
                    continue
            df = df.append({"ImageId":id_,
                            "EncodedPixels": run_length_encoder(msk == i),
                           },
                           ignore_index = True,
                          )
        return df
    
    print("Encoding pixels...")
    for id_, msk in tqdm(image_list):
        DF = run_length_writer(DF, id_, msk)
        
    print("Saving {}.".format(csv_file_path))
    DF.to_csv(csv_file_path, index = False)
    
def mask_store_from_rle(dir_path, csv):
    """Converts run length encoded string into an image and stores in dir_path
    Args:
        dir_path: directory where masked image will be stored
        csv: a csv file with header: "ImageId", "EncodedPixels", "Height","Width" 
    """
    import itertools
    import imageio
    from pathlib import Path
    
    def rle_inv(rle_str, H, W):
        """Turns run length encoding into a mask
        Args:
            rle_str: run length encoded string
            H: height of image
            W: width of image
        Returns:
            mask of shape (H,W) 
        """
        rle = [int(i) for i in rle_str.split(" ")]
        rle=[list(range(rle[i] - 1, rle[i] + rle[i+1] - 1)) for i in range(0, len(rle), 2)]
        index = list(itertools.chain.from_iterable(rle))
        msk = np.zeros(H * W, dtype = np.uint8)
        msk[index] = 255
        msk = msk.reshape((W,H)).T
        return msk

    for index in csv.index:
        id_ = csv["ImageId"][index]
        msk = rle_inv(csv["EncodedPixels"][index], 
                      csv["Height"][index],
                      csv["Width"][index],
                      )
        mskpath = os.path.join(dir_path, id_, "masks")
        Path(mskpath).mkdir(exist_ok = True)
        path = os.path.join(mskpath, str(index) + "_" + id_ + ".png")
        imageio.imwrite(path, msk)
   