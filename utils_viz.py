
import argparse
from math import degrees
import pickle
from tkinter import image_names

import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch

import imageio


import utils_vox
from utils import get_mesh_renderer, get_device, get_points_renderer


def visualize_voxel(voxels, image_size=256, voxel_size=64, device=None, azimuth_angle = 180):
    if device is None:
        device = get_device()
    min_value = -1.1
    max_value = 1.1
    # temp
    # X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    # print(X.shape)
    # print(X[1][2][3])
    # voxels = X ** 2 + Y ** 2 + Z ** 2 - 1
    # print(voxels[0][0][0])
    # print(voxels.shape, "vox before marching cube")
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    # print(vertices.shape, "vert after marching cube")
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    # print(vertices.shape, "before")
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    # print(vertices.shape, "after")
    # textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    # print(textures.shape)
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=4, elev=0, azim=azimuth_angle)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)

def visualize_point_cloud(verts, rgb,
    image_size=256,
    background_color=(1, 1, 1),
    azimuth_angle = 180,
    device=None):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    # point_cloud = np.load(point_cloud_path)
    # verts = torch.Tensor(point_cloud["verts"][::50]).to(device).unsqueeze(0)
    # rgb = torch.Tensor(point_cloud["rgb"][::50]).to(device).unsqueeze(0)
    verts = verts.to(device)
    rgb = rgb.to(device)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(3, 0, azimuth_angle,degrees = True)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    return rend

def visualize_mesh(vertices, faces, image_size=256, color=[0.7, 0.7, 1], device=None, azimuth_angle = 180):
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size,device= device)
    vertices = vertices.to(device).unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.to(device).unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices).to(device)  # (1, N_v, 3)
    color=[0.7, 0.7, 1]
    textures = textures * torch.tensor(color).to(device)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 3, -3]], device=device)
    r, t = pytorch3d.renderer.cameras.look_at_view_transform(dist = 4, elev = 0.0, azim = azimuth_angle, degrees= True)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=r, T=t, fov=60, device=device
    )
    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    return rend    

def visualize(data = None, vertices = None, faces = None, type= None, image_size = 512,voxel_size=32, device=None):

    image_list = []
    for i in range(0,360,4):
        # Prepare the camera:
        # r, t = pytorch3d.renderer.cameras.look_at_view_transform(dist = 3, elev = 0.0, azim = i, degrees= True)
        
    
        if type == "voxel":
            rend = visualize_voxel(data, image_size=image_size, voxel_size=voxel_size, device=device, azimuth_angle=i)
        if type == "point_cloud":
            verts = data
            color=[0.7, 0.7, 1]
            rgbs = torch.ones_like(verts,device = device) * torch.tensor(color, device = device)
            # rgbs = rgbs.to(device)
            rend = visualize_point_cloud(verts=verts,rgb=rgbs, image_size=image_size,device=device, azimuth_angle=i )
        if type == "mesh":
            rend = visualize_mesh(vertices=vertices, faces=faces, image_size=image_size, device=device, azimuth_angle=i)
        # rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        image_list.append(rend)

    return image_list



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="implicit",
        choices=["point_cloud", "voxel", "mesh"],
    )
    parser.add_argument("--output_path", type=str, default="utils_sphere.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--type", type=str,default= "image", choices = ["image","gif"])
    args = parser.parse_args()
    vertices = torch.tensor([[0.0,0.0,0.0],
                             [0.0,1.0,1.0],
                             [1.0,1.0,0.0],
                             [1.0,0.0,1.0]])
    faces = torch.tensor([[0,1,2],
                          [0,1,3],
                          [0,2,3],
                          [1,2,3]])
    
    
    if args.render == "mesh":
        image_list = visualize(vertices=vertices, faces=faces, type="mesh")
    # if args.render == "point_cloud":


    # if args.render == "implicit":
    #     image = visualize_mesh(0,image_size=args.image_size)
    # else:
    #     raise Exception("Did not understand {}".format(args.render))
    if(args.type == "image"):
        plt.imsave(args.output_path, image)
    if(args.type == "gif"):
        imageio.mimsave(args.output_path, image_list, fps=20)

