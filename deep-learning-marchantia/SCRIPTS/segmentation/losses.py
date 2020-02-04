from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import numpy as np

def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)

    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              ((iflat*iflat).sum() + (tflat*tflat).sum() + smooth))

def discriminative_loss(prediction, target, feature_dim, image_shape, delta_v, delta_d, usegpu):

    alpha = beta = 1.0
    gamma = 0.001
    param_scale = 1.

    device = torch.device('cuda:0')

    batch_size = prediction.size()[0]

    # Reshape so pixels are aligned along a vector
    prediction_flat = prediction.view(batch_size, feature_dim, image_shape[0]*image_shape[1])
    target_flat = target.view(batch_size, image_shape[0]*image_shape[1])

    total_loss = 0

    for img_i in range(batch_size):

        # Count instances
        unique_labels, unique_id = torch.unique(target_flat[img_i], return_inverse=True)
        unique_counts = torch.stack([(target_flat[img_i]==label).sum() for label in unique_labels])
        unique_counts = unique_counts.type(torch.cuda.FloatTensor)
        n_instances = unique_labels.shape[0]

        unique_id_expand = unique_id.repeat(feature_dim,1)

        # Calculate means (centres of clusters)
        segmented_sum = torch.zeros(feature_dim, n_instances)
        segmented_sum = segmented_sum.cuda(device)
        segmented_sum = segmented_sum.scatter_add(1, unique_id_expand, prediction_flat[img_i])

        means = torch.div(segmented_sum, unique_counts)
        means_expand = torch.gather(means, 1, unique_id_expand)

        ### Calculate variance term l_var
        distance = torch.norm((means_expand - prediction_flat[img_i]), dim=0) - delta_v
        distance = torch.clamp(distance, min=0.0) ** 2

        l_var = torch.zeros(n_instances)
        l_var = l_var.cuda(device)
        l_var = l_var.scatter_add(0, unique_id, distance)

        l_var = torch.div(l_var, unique_counts)
        l_var = torch.sum(l_var)
        l_var = torch.div(l_var, float(n_instances))

        ### Calculate distance term l_dist

        # Get distance for each pair of clusters like this:
        #   mu_1 - mu_1
        #   mu_2 - mu_1
        #   mu_3 - mu_1
        #   mu_1 - mu_2
        #   mu_2 - mu_2
        #   mu_3 - mu_2
        #   mu_1 - mu_3
        #   mu_2 - mu_3
        #   mu_3 - mu_3
        
        mu_interleaved_rep = torch.t(means).repeat(1,n_instances).view(-1, feature_dim)
        mu_band_rep = means.repeat(1, n_instances)
        mu_band_rep = torch.t(mu_band_rep)
        
        mu_diff = torch.add(mu_band_rep, -mu_interleaved_rep) # should be sized n_instances*n_instances

        # Filter out zeros from same cluster subtraction
        intermediate_tensor = torch.sum(torch.abs(mu_diff), dim=1)
        zero_vector = torch.zeros(1)
        zero_vector = zero_vector.cuda(device)
        bool_mask = torch.ne(intermediate_tensor, zero_vector)
        bool_mask = bool_mask.view(-1, 1)
        
        mu_diff_bool = torch.masked_select(mu_diff, bool_mask).view(-1,feature_dim)
        
        mu_norm = 2 * delta_d - torch.norm(mu_diff_bool, dim=1, p=2)
        mu_norm = torch.clamp(mu_norm, min=0.0) ** 2

        l_dist = torch.mean(mu_norm)

        ### Calculate regularization term l_reg
        l_reg = torch.mean(torch.norm(means, dim=1))

        loss = param_scale * (alpha * l_var + beta * l_dist + gamma * l_reg)

        total_loss = total_loss + loss
    
    return total_loss / batch_size

class DiscriminativeLoss(_Loss):

    def __init__(self, delta_var, delta_dist, image_shape, feature_dim, usegpu=True):
        super(DiscriminativeLoss, self).__init__(True)

        self.delta_var = float(delta_var)
        self.delta_dist = float(delta_dist)
        self.image_shape = image_shape
        self.feature_dim = feature_dim
        self.usegpu = usegpu

    def forward(self, input, target):
        loss = discriminative_loss(input, target, self.image_shape, self.feature_dim, self.delta_var, self.delta_dist, self.usegpu)

        return loss