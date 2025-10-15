import numpy as np
import os
from util import *
import argparse


def set_requires_grad(tensor_list):
    for tensor in tensor_list:
        tensor.requires_grad = True


parser = argparse.ArgumentParser()

parser.add_argument(
    "--path", type=str, default="", help="idname of target person")
parser.add_argument('--img_h', type=int, default=512, help='height if image')
parser.add_argument('--img_w', type=int, default=512, help='width of image')
args = parser.parse_args()
id_dir = args.path

# Try to load track_params.pt or track_params_deca.pt
track_params_path = os.path.join(id_dir, 'track_params.pt')
track_params_deca_path = os.path.join(id_dir, 'track_params_deca.pt')

if os.path.exists(track_params_path):
    params_dict = torch.load(track_params_path)
    print(f'Loaded parameters from {track_params_path}')
elif os.path.exists(track_params_deca_path):
    params_dict = torch.load(track_params_deca_path)
    print(f'Loaded parameters from {track_params_deca_path}')
else:
    raise FileNotFoundError(f'Neither {track_params_path} nor {track_params_deca_path} found. Please run face tracking (task 7) first.')

# Preserve face model type if available
face_model_type = params_dict.get('face_model_type', 'BFM')  # Default to BFM for backward compatibility
print(f'Face model type: {face_model_type}')

euler_angle = params_dict['euler'].cuda()
trans = params_dict['trans'].cuda() / 1000.0
focal_len = params_dict['focal'].cuda()

track_xys = torch.as_tensor(
    np.load(os.path.join(id_dir, 'track_xys.npy'))).float().cuda()
num_frames = track_xys.shape[0]
point_num = track_xys.shape[1]

pts = torch.zeros((point_num, 3), dtype=torch.float32).cuda()
set_requires_grad([euler_angle, trans, pts])

cxy = torch.Tensor((args.img_w/2.0, args.img_h/2.0)).float().cuda()

optimizer_pts = torch.optim.Adam([pts], lr=1e-2)
iter_num = 500
for iter in range(iter_num):
    proj_pts = forward_transform(pts.unsqueeze(0).expand(
        num_frames, -1, -1), euler_angle, trans, focal_len, cxy)
    loss = cal_lan_loss(proj_pts[..., :2], track_xys)
    optimizer_pts.zero_grad()
    loss.backward()
    optimizer_pts.step()


optimizer_ba = torch.optim.Adam([pts, euler_angle, trans], lr=1e-4)


iter_num = 8000
for iter in range(iter_num):
    proj_pts = forward_transform(pts.unsqueeze(0).expand(
        num_frames, -1, -1), euler_angle, trans, focal_len, cxy)
    loss_lan = cal_lan_loss(proj_pts[..., :2], track_xys)
    loss = loss_lan
    optimizer_ba.zero_grad()
    loss.backward()
    optimizer_ba.step()

torch.save({'euler': euler_angle.detach().cpu(),
            'trans': trans.detach().cpu(),
            'focal': focal_len.detach().cpu(),
            'face_model_type': face_model_type}, os.path.join(id_dir, 'bundle_adjustment.pt'))
print(f'bundle adjustment params saved (face model: {face_model_type})')
