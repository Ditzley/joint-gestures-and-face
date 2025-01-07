import os
import torch
import smplx as smpl
import numpy as np
import trimesh

from data_utils.show_data_utils.lower_body import part2full

def get_vertices(smplx_model, betas, result_list, exp, require_pose=False, face=False):
    vertices_list = []
    poses_list = []
    expression = torch.zeros([1, 100], device=betas.device)

    if len(betas.shape) == 1:
        betas = betas.unsqueeze(dim=0)

    for i in result_list:
        vertices = []
        poses = []
        for j in range(i.shape[0]):
            output = smplx_model(betas=betas,
                                 expression=i[j][165:265].unsqueeze(dim=0) if exp else expression,
                                 jaw_pose=i[j][0:3].unsqueeze(dim=0),
                                 leye_pose=i[j][3:6].unsqueeze(dim=0),
                                 reye_pose=i[j][6:9].unsqueeze(dim=0),
                                 global_orient=i[j][9:12].unsqueeze(dim=0),
                                 body_pose=i[j][12:75].unsqueeze(dim=0) if not face else torch.zeros_like(i[j][12:75].unsqueeze(dim=0), device=betas.device),
                                 left_hand_pose=i[j][75:120].unsqueeze(dim=0) if not face else torch.zeros_like(i[j][75:120].unsqueeze(dim=0), device=betas.device),
                                 right_hand_pose=i[j][120:165].unsqueeze(dim=0) if not face else torch.zeros_like(i[j][120:165].unsqueeze(dim=0), device=betas.device),
                                 return_verts=True)
            vertices.append(output.vertices.detach().cpu().numpy().squeeze())
            # trimesh.Trimesh(vertices=vertices[0], faces=smplx_model.faces).show()
            
            pose = output.body_pose
            poses.append(pose.detach().cpu())
        vertices = np.asarray(vertices)
        vertices_list.append(vertices)
        poses = torch.cat(poses, dim=0)
        poses_list.append(poses)
    if require_pose:
        return vertices_list, poses_list
    else:
        return vertices_list, None


class Renderer():
    def __init__(self, path=None):
        with torch.no_grad():
            smplx_path = './visualise/'
            dtype = torch.float32
            model_params = dict(model_path=smplx_path,
                                model_type='smplx',
                                create_global_orient=True,
                                create_body_pose=True,
                                create_betas=True,
                                num_betas=300,
                                create_left_hand_pose=True,
                                create_right_hand_pose=True,
                                use_pca=False,
                                flat_hand_mean=False,
                                create_expression=True,
                                num_expression_coeffs=100,
                                num_pca_comps=12,
                                create_jaw_pose=True,
                                create_leye_pose=True,
                                create_reye_pose=True,
                                create_transl=False,
                                # gender='ne',
                                dtype=dtype, )
            self.smplx_model = smpl.create(**model_params)

    def render(self, method, face_pred, body_pred, face_gt, body_gt, audio_file, betas):
        pred = torch.cat([face_pred[:, :3], body_pred, face_pred[:, -100:]], dim=-1)
        gt = torch.cat([face_gt[:, :3], body_gt, face_gt[:, -100:]], dim=-1)
        gt = gt[:pred.shape[0]]

        face = False
        stand = False
        pred = part2full(pred, stand)
        gt = part2full(gt, stand)

        result_list = []
        result_list.append(gt[:248])
        result_list.append(pred[:248])

        vertices_list, _ = get_vertices(self.smplx_model.to(pred.device), betas, result_list, True)
        dict = np.concatenate(vertices_list[1:], axis=0)
        file_name = 'visualise/' + method + '/' + \
                    audio_file.split('/')[-4].split('.')[-2] + '/' + \
                    audio_file.split('/')[-1].split('.')[-2].split('/')[-1]

        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        np.save(file_name, dict)

        dict = pred[:248].cpu().numpy()
        file_name = 'visualise/' + method + '/' + \
                    audio_file.split('/')[-4].split('.')[-2] + '/joints_' + \
                    audio_file.split('/')[-1].split('.')[-2].split('/')[-1]
        np.save(file_name, dict)

    def view(self, face, body, betas):
        pose = torch.cat([face[:, :3], body, face[:, -100:]], dim=-1)
        pose = part2full(pose, False)

        vertices_list, _ = get_vertices(self.smplx_model.to(pose.device), betas, [pose], True)
        for i in vertices_list[0]:
            mesh = trimesh.Trimesh(i, self.smplx_model.faces)
            mesh.show()
