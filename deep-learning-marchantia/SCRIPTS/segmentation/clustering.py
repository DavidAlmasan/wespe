import os
import numpy as np
from sklearn.cluster import MeanShift
import time

import random

def cluster(prediction, bandwidth=1.):
    """
    Parameters:
        prediction (array): array of pixels in feature space, [n_pixels, n_features]
        bandwidth (float, optional): bandwidth of RBF kernel for meanshift algorithm
    """
    ms = MeanShift(bandwidth, bin_seeding=True)

    print ('Mean shift clustering, might take some time ...')
    tic = time.time()
    ms.fit(prediction)
    print ('time for clustering', time.time() - tic)

    labels = ms.labels_ # label for each pixel: array, [n_pixels]
    cluster_centers = ms.cluster_centers_ # coordinates of cluster centers: array, [n_clusters, n_features]
    
    labels_unique = np.unique(labels)
    num_clusters = len(labels_unique)

    return num_clusters, labels, cluster_centers

def get_instance_masks(prediction, bandwidth=1.):

    batch_size, feature_dim, h, w = prediction.shape
    print('prediction shape: '+ str(prediction.shape))
    print('feature_dim: ' + str(feature_dim))

    color_list = np.load('./color_index_palette.npy')

    instance_masks = []
    instance_index_masks = []
    for i in range(batch_size):

        random.seed(100)

        num_clusters, labels, cluster_centers = cluster(prediction[i].reshape([feature_dim, h*w]).t(), bandwidth)
        print('Number of predicted clusters: ', num_clusters)
        labels = np.array(labels.reshape([h,w]), dtype=np.uint32)
        print('labels: ' + str(labels.shape))
        print('unique labels: ', len(np.unique(labels)))
        mask = np.zeros([h,w,3], dtype=np.uint8) # contains color mask for each label
        print('mask: ' + str(mask.shape))
        index_mask = np.zeros([h,w], dtype=np.uint32) # contains index mask for each label

        for mask_id in range(num_clusters):
            ind = np.where(labels==mask_id)
            #print(ind)
            [r,g,b] = color_list[mask_id,:]
            mask[ind] = np.array([r,g,b])
            index_mask[ind] = mask_id
        
        instance_masks.append(mask)
        instance_index_masks.append(index_mask)
    
    return instance_masks, instance_index_masks





