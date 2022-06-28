from base64 import encode
from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d

class Reshape(nn.Module):
    def __init__(self,shape):
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.reshape(self.shape)


class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = "cuda"
        vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
        self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        # define decoder
        if args.type == "vox":
            # TODO:
            self.decoder =  torch.nn.Sequential(nn.Linear(512,32*32*32),
                                                Reshape((-1,1,32,32,32)))
                                                # nn.GELU(),
                                                # nn.Linear(1024,2048),
                                                # nn.GELU(),
                                                # nn.Linear(2048,32*32*32))

        elif args.type == "point":
            self.n_point = args.n_points
            # TODO:
            self.decoder = self.decoder =  torch.nn.Sequential(nn.Linear(512,1024),
                                                               nn.GELU(),
                                                               nn.Linear(1024,args.n_points*3),
                                                               Reshape((-1,args.n_points, 3))
                                                            )
                                                # nn.Linear(1024,2048),
                                                # nn.GELU(),
                                                # nn.Linear(2048,32*32*32))
                                                
        elif args.type == "mesh":
            # try different mesh initializations
            mesh_pred = ico_sphere(4,'cuda')
            self.n_vertices = mesh_pred.verts_packed().size(0)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            # TODO:
            # self.num_vertices = len(self.mesh_pred.verts_list())
            # self.num_faces = len(self.mesh_pred.faces_list())
            self.decoder = torch.nn.Sequential(nn.Linear(512,1024),
                                                nn.GELU(),
                                                nn.Linear(1024,2048),
                                                nn.GELU(),
                                                nn.Linear(2048, self.n_vertices*3),
                                                Reshape((-1,self.n_vertices,3)))      

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        images_normalize = self.normalize(images.permute(0,3,1,2))
        encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1)

        #print the dimensions of the output of the encoder
        # print(encoded_feat.shape)
        # call decoder
        if args.type == "vox":
            # TODO:
            # voxels_pred =  self.decoder(encoded_feat).view(-1,1,32,32,32)
            voxels_pred =  self.decoder(encoded_feat)         
            # print(voxels_pred.shape)
            return voxels_pred

        elif args.type == "point":
            # TODO:
            pointclouds_pred = self.decoder(encoded_feat).view(-1,args.n_points,3)            
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            deform_vertices_pred = self.decoder(encoded_feat)            
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          

