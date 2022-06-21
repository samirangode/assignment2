import argparse
import os
import time

import losses
from pytorch3d.utils import ico_sphere
from r2n2_custom import R2N2
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
import dataset_location
import torch
import mcubes
import utils_viz
import utils
import imageio


import matplotlib.pyplot as plt




def get_args_parser():
    parser = argparse.ArgumentParser('Model Fit', add_help=False)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--max_iter', default=100000, type=int)
    parser.add_argument('--log_freq', default=1000, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh', 'test'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    # parser.add_argument('--output_path', type = str)
    return parser

def fit_mesh(mesh_src, mesh_tgt, args):
    start_iter = 0
    start_time = time.time()

    deform_vertices_src = torch.zeros(mesh_src.verts_packed().shape,requires_grad=True,device='cuda')
    optimizer = torch.optim.Adam([deform_vertices_src], lr = args.lr)
    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        new_mesh_src = mesh_src.offset_verts(deform_vertices_src)

        sample_trg = sample_points_from_meshes(mesh_tgt, args.n_points)
        sample_src = sample_points_from_meshes(new_mesh_src, args.n_points)

        loss_reg = losses.chamfer_loss(sample_src, sample_trg)
        loss_smooth = losses.smoothness_loss(new_mesh_src)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))        
    
    mesh_src.offset_verts_(deform_vertices_src)

    print('Done!')


    



def fit_pointcloud(pointclouds_src, pointclouds_tgt, args):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([pointclouds_src], lr = args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.chamfer_loss(pointclouds_src, pointclouds_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    print('Done!')


def fit_voxel(voxels_src, voxels_tgt, args):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([voxels_src], lr = args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.voxel_loss(voxels_src,voxels_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    print('Done!')


def train_model(args):
    r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)

    
    feed = r2n2_dataset[0]


    feed_cuda = {}
    for k in feed:
        if torch.is_tensor(feed[k]):
            feed_cuda[k] = feed[k].cuda().float()


    if args.type == "vox":
        # initialization
        voxels_src = torch.rand(feed_cuda['voxels'].shape,requires_grad=True,device='cuda')
        voxel_coords = feed_cuda['voxel_coords'].unsqueeze(0)
        voxels_tgt = feed_cuda['voxels']
        # print("###################")
        # print(voxels_src, "voxels_src")
        # print("####################")
        # print(voxels_tgt, "voxels_tgt")
        voxels_src = voxels_src.cpu().detach().squeeze(0)
        voxels_tgt = voxels_tgt.cpu().detach().squeeze(0)
        # print(voxels_src.shape)
        # print(voxels_tgt.shape)
        # image_src = utils_viz.visualize_mesh(voxels_src,voxel_size= voxels_src.shape[0])
        # plt.imshow(image_src)
        # image_tgt = utils_viz.visualize_mesh(voxels_tgt, voxel_size= voxels_tgt.shape[0])
        image_list = utils_viz.visualize(data=voxels_tgt,type="voxel", device = utils.get_device())
        imageio.mimsave("voxel_tgt.gif", image_list, fps=20)

        
        # plt.imshow(image_tgt)
        # plt.show()
        # # fitting
        # vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
        # vertices = torch.tensor(vertices).float()
        # faces = torch.tensor(faces.astype(int))
        # # Vertex coordinates are indexed by array position, so we need to
        # # renormalize the coordinate system.
        # vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
        # textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
        # textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

        # mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        #     device
        # )
        # lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
        # renderer = get_mesh_renderer(image_size=image_size, device=device)
        # R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
        # cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        # rend = renderer(mesh, cameras=cameras, lights=lights)
        
        ########################
        voxels_src = torch.rand(feed_cuda['voxels'].shape,requires_grad=True,device='cuda')
        voxel_coords = feed_cuda['voxel_coords'].unsqueeze(0)
        voxels_tgt = feed_cuda['voxels']
        fit_voxel(voxels_src, voxels_tgt, args)
        voxels_src = voxels_src.cpu().detach().squeeze(0)
        # image_src_optimized = utils_viz.visualize_mesh(voxels_src)
        # plt.imshow(image_src_optimized)
        # plt.show()
        image_list = utils_viz.visualize(data=voxels_src,type="voxel", device = utils.get_device())
        imageio.mimsave("voxel_src.gif", image_list, fps=20)
        print(utils.get_device())


    elif args.type == "point":
        # initialization
        pointclouds_src = torch.randn([1,args.n_points,3],requires_grad=True,device='cuda')
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])
        pointclouds_tgt = sample_points_from_meshes(mesh_tgt, args.n_points)
        # fitting
        # print(pointclouds_tgt.shape)
        # for i in pointclouds_tgt:
        #     print(i)
        # pointclouds_tgt = pointclouds_tgt.squeeze(0)
        # print(pointclouds_tgt["verts"].shape)
        # print(pointclouds_tgt.shape)
        
        image_list = utils_viz.visualize(data=pointclouds_tgt[::50],type="point_cloud", device = utils.get_device())
        imageio.mimsave("point_cloud_tgt.gif", image_list, fps=20)
        # pointclouds_tgt = pointclouds_tgt.unsqueeze(0)

        fit_pointcloud(pointclouds_src, pointclouds_tgt, args)
        # pointclouds_src = pointclouds_src.squeeze(0)
        image_list = utils_viz.visualize(data=pointclouds_src[::50].detach(),type="point_cloud", device = utils.get_device())
        imageio.mimsave("point_cloud_src.gif", image_list, fps=20)        
        # pointclouds_src = pointclouds_src.unsqueeze(0)

    elif args.type == "mesh":
        # initialization
        # try different ways of initializing the source mesh        
        mesh_src = ico_sphere(4,'cuda')
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])

        # fitting
        image_list = utils_viz.visualize(vertices=mesh_tgt.verts_packed(), faces=mesh_tgt.faces_packed(),type="mesh", image_size=512,device=utils.get_device())
        imageio.mimsave("mesh_tgt.gif", image_list, fps = 20)
        fit_mesh(mesh_src, mesh_tgt, args)
        image_list = utils_viz.visualize(vertices=mesh_src.detach().verts_packed(), faces=mesh_src.detach().faces_packed(),type="mesh", image_size=512,device=utils.get_device())
        imageio.mimsave("mesh_src.gif", image_list, fps = 20)

    elif args.type == "test":
        #get the maximum
        voxels_src = torch.rand(feed_cuda['voxels'].shape,requires_grad=True,device='cuda')
        print(voxels_src.shape)
        print(voxels_src[0])
        print(torch.max(voxels_src,dim = 1))
        print(torch.max(voxels_src,dim = 2))
        print(torch.max(voxels_src,dim = 3))





    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Fit', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
