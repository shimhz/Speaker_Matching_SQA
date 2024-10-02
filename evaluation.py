import numpy as np
from tqdm import tqdm
import torch
import pickle as pk
import scipy


def convert_and_concatenate(lm_dict, spk_dict, key):
    # Convert numpy arrays to PyTorch tensors
    tensor_lm = torch.from_numpy(lm_dict[key])
    tensor_spk = torch.from_numpy(spk_dict[key])

    # Flatten tensor_lm along the last two dimensions
    tensor_lm_flattened = tensor_lm.view(tensor_lm.shape[0], -1)  # shape: (1, 449 * 1024)

    # Ensure tensor_spk is 2D to concatenate with tensor_lm_flattened
    tensor_spk_expanded = tensor_spk.unsqueeze(0)  # shape: (1, 192)

    # Concatenate the flattened tensor_lm with the expanded tensor_spk along the last dimension (dim=1)
    concatenated_tensor = torch.cat((tensor_lm_flattened, tensor_spk_expanded), dim=1)
    
    return concatenated_tensor

def calculate_bert(v_generated, v_reference):
    """
    Args:
        v_generated (torch.Tensor): Generated feature tensor (T, D).
        v_reference (torch.Tensor): Reference feature tensor (T, D).
    Returns:
        float: Precision.
        float: Recall.
        float: F1 score.
    """

    # Calculate cosine similarity
    sim_matrix = torch.matmul(v_generated, v_reference.T) / (torch.norm(v_generated, dim=1, keepdim=True) * torch.norm(v_reference, dim=1).unsqueeze(0))

    # Calculate precision and recall
    precision = torch.max(sim_matrix, dim=1)[0].mean().item()
    #recall = torch.max(sim_matrix, dim=0)[0].mean().item()
    #f1_score = 2 * precision * recall / (precision + recall)

    #return precision, recall, f1_score
    return precision

            
def calculate_correlations(l_score, l_mos):
    SRCC_rho, SRCC_pval = scipy.stats.spearmanr(l_score, l_mos)
    LCC = scipy.stats.pearsonr(l_score, l_mos)

    print ("SRCC Rho: {}, SRCC p-value: {}\n".format(SRCC_rho, SRCC_pval))
    print ("LCC :{}\n".format(LCC))
    return SRCC_rho, LCC

