import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds
from utils import get_device
from pytorch3d.loss import mesh_laplacian_smoothing

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	loss =  torch.nn.BCEWithLogitsLoss()
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
	cham_x = cham_x.sum()  # (N,)
	cham_y = cham_y.sum()  # (N,)
	loss_chamfer = cham_x + cham_y
	# implement chamfer loss from scratch
	# print(loss_chamfer, "loss chamfer")
	return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = mesh_laplacian_smoothing(mesh_src)
	# # implement laplacian smoothening loss
	# # print(loss_laplacian, "loss laplacian")
	# return loss_laplacian

	V = mesh_src.verts_packed()
	L = mesh_src.laplacian_packed()
	
	# faces = mesh_src.faces_packed()
	# L = torch.zeros((V.size(0), V.size(0)))

	# # for face in faces:
	# L[faces[:, 0], faces[:, 0]] += 1
	# L[faces[:, 0], faces[:, 1]] -= 1
	# L[faces[:, 0], faces[:, 2]] -= 1
	# L[faces[:, 1], faces[:, 0]] -= 1
	# L[faces[:, 1], faces[:, 1]] += 1
	# L[faces[:, 1], faces[:, 2]] -= 1
	# L[faces[:, 2], faces[:, 0]] -= 1
	# L[faces[:, 2], faces[:, 1]] -= 1
	# L[faces[:, 2], faces[:, 2]] += 1

	loss_laplacian = torch.square(torch.linalg.norm(torch.matmul(L, V)))
		# implement laplacian smoothening loss
	return loss_laplacian