import os
import sys
import random

from jsonargparse import ActionConfigFile

sys.path.append(os.getcwd())

import lightning as L
from lightning.pytorch.strategies import DDPStrategy
import lightning.pytorch.cli as cli
from lightning.pytorch.callbacks import ModelCheckpoint

from scripts.nets.diffuser import Diffuser as JointDiffuser
from data_utils.gesture_datasets import get_gesture_dataset
from data_utils.show_data_utils.lower_body import *

import smplx as smpl

torch.set_float32_matmul_precision('high')

os.environ['smplx_npz_path'] = "visualise/smplx_model/SMPLX_NEUTRAL_2020.npz"
os.environ['extra_joint_path'] = "visualise/smplx_model/smplx_extra_joints.yaml"
os.environ['j14_regressor_path'] = "visualise/smplx_model/SMPLX_to_J14.pkl"

def create_model(device='cuda'):
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
                        dtype=dtype,)
    smplx_model = smpl.create(**model_params).to(torch.device(device))
    return smplx_model

if __name__ == '__main__':
    parser = cli.LightningArgumentParser()
    parser.add_lightning_class_args(JointDiffuser, 'model')
    parser.add_lightning_class_args(L.Trainer, 'trainer')
    parser.add_function_arguments(get_gesture_dataset, 'data')

    parser.add_argument('-c', '--config', action=ActionConfigFile, help="Path to a configuration file in json or yaml format.")
    # parser.link_arguments("trainer.accumulate_grad_batches", "model.accumulate_grad_batches")
    # parser.link_arguments("trainer.gradient_clip_val", "model.gradient_clip_val")
    # parser.link_arguments("trainer.gradient_clip_algorithm", "model.gradient_clip_algorithm")

    parser.link_arguments("model.num_poses", "data.num_frames")

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--infer', action='store_true')

    parser.add_argument('--model_path', type=str, default=None)

    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.link_arguments("exp_name", "model.method_name")
    parser.add_argument('--logger', type=str, default=None)

    config = parser.parse_args(defaults=True)

    config.trainer.callbacks = [
        ModelCheckpoint(
            filename='{epoch}',
            every_n_epochs=50,
            save_last=True,
            save_top_k=-1
        )
    ]

    if config.seed is not None:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True

    if not config.train:
        config.trainer.devices = 1
        config.trainer.strategy = 'auto'

    if config.trainer.strategy == 'ddp':
        config.trainer.strategy = DDPStrategy()

    # init trainer
    trainer = L.Trainer(
        **config.trainer,
    )
    model = JointDiffuser(
        **config.model
    )

    datamodule = get_gesture_dataset(
        **config.data
    )

    if config.train:
        trainer.fit(model, datamodule=datamodule, ckpt_path=config.model_path)
    elif config.infer:
        trainer.predict(model, datamodule=datamodule, ckpt_path=config.model_path)