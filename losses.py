import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds
from utils import get_device
from pytorch3d.loss import mesh_laplacian_smoothing

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	loss = torch.nn.CrossEntropyLoss()
	# implement some loss for binary voxel grids
	prob_loss = loss(voxel_src, voxel_tgt)
	return prob_loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	x = point_cloud_src
	y = point_cloud_tgt
	x_len = torch.tensor(x.shape[1],dtype = torch.int64, device=x.device)
	y_len = torch.tensor(x.shape[1],dtype = torch.int64, device=y.device)
	x_nn = knn_points(x, y, lengths1=x_len, lengths2=y_len, K=1)
	y_nn = knn_points(y, x, lengths1=y_len, lengths2=x_len, K=1)
	
	cham_x = x_nn.dists[..., 0]  # (N, P1)
	cham_y = y_nn.dists[..., 0]  # (N, P2)
	# Apply point reduction
	cham_x = cham_x.sum(1)  # (N,)
	cham_y = cham_y.sum(1)  # (N,)
	loss_chamfer = cham_x + cham_y
	# implement chamfer loss from scratch
	return loss_chamfer

def smoothness_loss(mesh_src):
	loss_laplacian = mesh_laplacian_smoothing(mesh_src)
	# implement laplacian smoothening loss
	return loss_laplacian