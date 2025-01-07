import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from tqdm import tqdm

from data_utils.show_data_utils.lower_body import c_index_3d
from .diffusion_util import VarianceSchedule

from .transformer_model import TransformerModel

from scripts.renderer import Renderer

class Diffuser(L.LightningModule):
    def __init__(self,
                 num_poses=88,
                 num_pre_poses=None,
                 prediction_target='sample',
                 use_class_labels=False,
                 predict_parts=['face', 'upper_body'],
                 audio_feat_dim=1024,
                 audio_map_dim=64,
                 input_dim=64,
                 num_hiddens=512,
                 num_hidden_layers=8,
                 num_steps=500,
                 classifier_free=False,
                 null_cond_prob=0.1,
                 do_annealing=False,
                 split=True,
                 adapter_reduction_factor=8,
                 method_name='test',
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.prediction_target = prediction_target
        self.method_name = method_name
        self.num_poses = num_poses
        self.num_pre_poses = num_pre_poses

        self.predict_face = True if 'face' in predict_parts else False
        self.predict_body = True if 'upper_body' in predict_parts else False
        self.lower_body = True if 'lower_body' in predict_parts else False

        self.audio_map_dim = audio_map_dim
        self.audio_feat_dim = audio_feat_dim

        self.num_hiddens = num_hiddens
        self.num_hidden_layers = num_hidden_layers

        self.use_class_labels = use_class_labels

        self.c_index = c_index_3d

        self.face_dim = 103
        self.hand_dim = 90
        self.upper_body_dim = 39
        self.lower_body_dim = 33
        self.body_dim = self.hand_dim + self.upper_body_dim

        self.input_dim = input_dim
        self.output_dim = input_dim

        self.audio_map = nn.Linear(self.audio_feat_dim, self.audio_map_dim)

        self.split = split

        self.face_to_latent = nn.Linear(self.face_dim, self.input_dim)
        self.latent_to_face = nn.Linear(self.output_dim, self.face_dim)

        self.body_to_latent = nn.Linear(self.body_dim, self.input_dim)
        self.latent_to_body = nn.Linear(self.output_dim, self.body_dim)

        input_dim = self.input_dim * 2 + 1 + self.audio_map_dim + 3 + (4 if self.use_class_labels else 0)

        self.model = TransformerModel(
            seq_len=self.num_poses,
            input_dim=input_dim,
            output_dim=self.output_dim,
            hidden_dim=self.num_hiddens,
            encoder_depth=self.num_hidden_layers // 2,
            encoder_heads=8,
            decoder_depth=self.num_hidden_layers // 2,
            decoder_heads=8,
            mlp_ratio=4.,
            norm_layer=nn.LayerNorm,
            split=split,
            adapter_reduction_factor=adapter_reduction_factor
        )

        self.noise_scheduler = VarianceSchedule(
            num_steps=num_steps,
            beta_1=1e-4,
            beta_T=0.02,
            mode='linear'
        )

        # diffusion sampling params
        self.do_annealing = do_annealing
        self.classifier_free = classifier_free

        if self.classifier_free:
            self.null_cond_prob = null_cond_prob
            null_embed_dim = self.input_dim * 2 + 1 * 2 + self.audio_map_dim * 2 + (8 if self.use_class_labels else 0)
            self.null_cond_emb = nn.Parameter(torch.randn(1, null_embed_dim))
        self.do_eval = False

    def get_pre_poses(self, poses):
        pre_pose = poses.new_zeros((poses.shape[0], poses.shape[1], poses.shape[2] + 1))
        pre_pose[:, 0:self.num_pre_poses, :-1] = poses[:, 0:self.num_pre_poses]
        pre_pose[:, 0:self.num_pre_poses, -1] = 1  # indicating bit for constraints
        return pre_pose

    def configure_optimizers(self):
        parameters = list(self.model.parameters())

        parameters += (list(self.face_to_latent.parameters())
                       + list(self.body_to_latent.parameters())
                       + list(self.latent_to_face.parameters())
                       + list(self.latent_to_body.parameters()))

        parameters += list(self.audio_map.parameters())

        optimizer = torch.optim.Adam(
            parameters,
            lr=1e-4,
            betas=(0.9, 0.999)
        )

        return optimizer

    def get_loss(self, target, pred, rec_weight=1.0, vel_weight=1.0):
        loss_dict = {}

        rec_mat = rec_weight * F.l1_loss(pred, target, reduction='none')
        rec_loss = torch.mean(rec_mat)
        loss_dict['rec_loss'] = rec_loss

        velocity_weight = vel_weight
        v_pr = pred[:, 1:] - pred[:, :-1]
        v_gt = target[:, 1:] - target[:, :-1]
        velocity_loss = torch.mean(torch.abs(v_pr - v_gt))
        vel_loss = velocity_weight * velocity_loss

        loss_dict['vel_loss'] = vel_loss

        loss_dict['loss'] = loss_dict['rec_loss'] + loss_dict['vel_loss']

        return loss_dict

    def get_inputs(self, batch):
        audio_feat = batch['hubert'].squeeze(dim=1)

        poses = batch['poses']

        # split pose to face/body
        face_poses = poses[:, :3, :]
        expression = batch['expression']
        face_poses = torch.cat([face_poses, expression], dim=1).permute(0, 2, 1)

        body_poses = poses[:, self.c_index, :].permute(0, 2, 1)
        class_labels = F.one_hot(batch['speaker'] - 20, 4)

        return face_poses, body_poses, audio_feat, class_labels

    def process_inputs(self, face_poses, body_poses, audio_feat, class_labels=None):
        face_latent = self.face_to_latent(face_poses)
        body_latent = self.body_to_latent(body_poses)

        face_pre = self.get_pre_poses(face_latent)
        body_pre = self.get_pre_poses(body_latent)

        audio_feat = self.audio_map(audio_feat)

        body_context = torch.cat([body_pre, audio_feat], dim=-1)
        face_context = torch.cat([face_pre, audio_feat], dim=-1)

        if class_labels is not None:
            class_labels = class_labels.unsqueeze(dim=1).repeat(1, self.num_poses, 1)
            body_context = torch.cat([body_context, class_labels], dim=-1)
            face_context = torch.cat([face_context, class_labels], dim=-1)

        context = torch.cat([face_context, body_context], dim=-1)

        joint_poses = torch.cat([face_poses, body_poses], dim=-1)

        return joint_poses, context

    def forward_train(self, x0, context):
        if self.classifier_free:
            mask = torch.zeros((x0.shape[0],), device=x0.device, dtype=x0.dtype).uniform_(0, 1) < self.null_cond_prob
            context = torch.where(mask.unsqueeze(1).unsqueeze(2), self.null_cond_emb.repeat(context.shape[1], 1).unsqueeze(0), context)

        bsz = x0.shape[0]
        e_rand = torch.randn_like(x0)

        timesteps = self.noise_scheduler.uniform_sample_t(bsz)
        alpha_bar = self.noise_scheduler.alpha_bars[timesteps]
        beta = self.noise_scheduler.betas[timesteps]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

        noisy_x0 = c0 * x0 + c1 * e_rand

        noisy_f = noisy_x0[:, :, :103]
        noisy_b = noisy_x0[:, :, 103:]
        face_latent = self.face_to_latent(noisy_f)
        body_latent = self.body_to_latent(noisy_b)

        noisy_sample = torch.cat([face_latent, body_latent], dim=-1)
        face_latent_pred, body_latent_pred = self.model(noisy_sample, beta, context)

        face_pred = self.latent_to_face(face_latent_pred)
        body_pred = self.latent_to_body(body_latent_pred)

        return face_pred, body_pred, face_latent_pred, body_latent_pred

    def training_step(self, batch, batch_idx):
        face_poses, body_poses, audio_feat, class_labels = self.get_inputs(batch)
        joint_poses, context = self.process_inputs(face_poses, body_poses, audio_feat,
                                                   class_labels=class_labels if self.use_class_labels else None)

        face_pred, body_pred, face_latent_pred, body_latent_pred = self.forward_train(joint_poses, context)

        # compute loss
        face_loss = self.get_loss(face_poses, face_pred, 1.0, 1.0)
        body_loss = self.get_loss(body_poses, body_pred, 1.0, 1.0)

        loss_dict = {}
        loss_dict['loss'] = face_loss['loss'] + body_loss['loss']
        loss_dict['face_rec'] = face_loss['rec_loss']
        loss_dict['face_vel'] = face_loss['vel_loss']
        loss_dict['body_rec'] = body_loss['rec_loss']
        loss_dict['body_vel'] = body_loss['vel_loss']

        self.log_dict(loss_dict, prog_bar=True)

        return loss_dict

    def validation_step(self, batch, batch_idx):
        pass

    def on_test_start(self):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def forward_sample(self, context):
        batch_size = 1
        x_t = torch.randn([batch_size, self.num_poses, self.face_dim + self.body_dim]).to(context.device)

        traj = []
        if self.classifier_free:
            uncondition_embedding = self.null_cond_emb.repeat(context.shape[1], 1).unsqueeze(0)
        else:
            uncondition_embedding = None

        for t in tqdm(range(self.noise_scheduler.num_steps, 0, -1), disable=True):
            z = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)

            alpha = self.noise_scheduler.alphas[t]
            alpha_bar = self.noise_scheduler.alpha_bars[t]
            sigma = self.noise_scheduler.get_sigmas(t, 0.0)
            beta = self.noise_scheduler.betas[[t] * batch_size]

            if uncondition_embedding is not None:
                noisy_f = x_t[:, :, :103]
                noisy_b = x_t[:, :, 103:]
                face_latent = self.face_to_latent(noisy_f)
                body_latent = self.body_to_latent(noisy_b)
                x_in = torch.cat([face_latent, body_latent], dim=-1)

                x_in = torch.cat([x_in] * 2)
                beta_in = torch.cat([beta] * 2)

                uncond_emb = uncondition_embedding.repeat(x_t.shape[0],1,1)
                context_in = torch.cat([uncond_emb, context])
                pred_face_latent, pred_body_latent = self.model(x_in, beta=beta_in, context=context_in)#.chunk(2)
                pred_face = self.latent_to_face(pred_face_latent)
                pred_body = self.latent_to_body(pred_body_latent)

                pred_uncond_out, pred_out = torch.cat([pred_face, pred_body], dim=-1).chunk(2)

                pred = pred_uncond_out + 1.15 * (pred_out - pred_uncond_out)

            else:
                noisy_f = x_t[:, :, :103]
                noisy_b = x_t[:, :, 103:]
                face_latent = self.face_to_latent(noisy_f)
                body_latent = self.body_to_latent(noisy_b)
                x_in = torch.cat([face_latent, body_latent], dim=-1)
                pred_face_latent, pred_body_latent = self.model(x_in, beta=beta, context=context)
                pred_face = self.latent_to_face(pred_face_latent)
                pred_body = self.latent_to_body(pred_body_latent)

                pred = torch.cat([pred_face, pred_body], dim=-1)
                traj.append(pred)

            if self.do_annealing:
                t0 = 25
                if t < t0 and t > 1:
                    sigma_a = 1/t0/t0*(t-t0)*(t-t0)
                    z0 = sigma_a * torch.randn_like(x_t[:,0,:].unsqueeze(1))

                    res = torch.zeros_like(z)
                    for n in range(x_t.shape[1]):
                        zn = math.sqrt((1-sigma_a*sigma_a)) * torch.randn_like(x_t[:,0,:].unsqueeze(1))
                        res[:,n:n+1,:] = zn + z0
                    z = res

            if self.prediction_target == 'sample':
                if t > 1:
                    pred_noise = (torch.sqrt(1.0 / alpha_bar) * x_t - pred) / torch.sqrt(1.0 / alpha_bar - 1.0)
                    alpha_next = self.noise_scheduler.alpha_bars[t-1]
                    c = torch.sqrt(1 - alpha_next - sigma ** 2)
                    x_t = pred * torch.sqrt(alpha_next) + c * pred_noise + sigma * z
                else:
                    x_t = pred

        pred_face = pred[:, :, :103]
        pred_body = pred[:, :, 103:]

        pred_face_in_latent = self.face_to_latent(pred_face)
        pred_body_in_latent = self.body_to_latent(pred_body)

        return pred_face, pred_body, pred_face_in_latent, pred_body_in_latent, traj

    def inference(self, batch, batch_idx):
        face_poses, body_poses, audio_feat, class_labels = self.get_inputs(batch)
        if self.use_class_labels:
            class_labels = class_labels.unsqueeze(dim=1).repeat(1, self.num_poses, 1)
        else:
            class_labels = None

        num_segments = int(face_poses.shape[1] // (self.num_poses - self.num_pre_poses))
        num_segments = num_segments - 1 if (num_segments * (self.num_poses - self.num_pre_poses) + self.num_pre_poses) > face_poses.shape[1] else num_segments

        face_latent = self.face_to_latent(face_poses[:, :self.num_poses])
        body_latent = self.body_to_latent(body_poses[:, :self.num_poses])

        face_pre = self.get_pre_poses(face_latent)
        body_pre = self.get_pre_poses(body_latent)

        face_pose_segments = []
        body_pose_segments = []
        face_clips = []
        body_clips = []

        audio_feat = self.audio_map(audio_feat)
        for i in range(num_segments):
            index = i * self.num_poses - i * self.num_pre_poses

            audio_ft = audio_feat[:, index:(index + self.num_poses)]
            if audio_ft.shape[1] < self.num_poses:
                continue

            face_context = torch.cat([face_pre, audio_ft], dim=-1)
            body_context = torch.cat([body_pre, audio_ft], dim=-1)
            if class_labels is not None:
                body_context = torch.cat([body_context, class_labels], dim=-1)
                face_context = torch.cat([face_context, class_labels], dim=-1)
            context = torch.cat([face_context, body_context], dim=-1)

            pred_face, pred_body, pred_face_latent, pred_body_latent, traj = self.forward_sample(context)

            face_clips.append(pred_face)
            body_clips.append(pred_body)

            face_pre[:, 0:self.num_pre_poses, :-1] = pred_face_latent[:, -self.num_pre_poses:, :]
            body_pre[:, 0:self.num_pre_poses, :-1] = pred_body_latent[:, -self.num_pre_poses:, :]

            if len(face_pose_segments) > 0:
                face_last_poses = face_pose_segments[-1][:, -self.num_pre_poses:]
                body_last_poses = body_pose_segments[-1][:, -self.num_pre_poses:]
                face_pose_segments[-1] = face_pose_segments[-1][:, :-self.num_pre_poses]  # delete last M frames
                body_pose_segments[-1] = body_pose_segments[-1][:, :-self.num_pre_poses]  # delete last M frames

                for j in range(face_last_poses.shape[1]):
                    n = face_last_poses.shape[1]
                    face_prev = face_last_poses[:, j]
                    face_next = pred_face[:, j]
                    pred_face[:, j] = face_prev * (n - j) / (n + 1) + face_next * (j + 1) / (n + 1)

                    n = body_last_poses.shape[1]
                    body_prev = body_last_poses[:, j]
                    body_next = pred_body[:, j]
                    pred_body[:, j] = body_prev * (n - j) / (n + 1) + body_next * (j + 1) / (n + 1)

            face_pose_segments.append(pred_face)
            body_pose_segments.append(pred_body)

        face_out_pose = torch.cat(face_pose_segments, dim=1)[0]
        body_out_pose = torch.cat(body_pose_segments, dim=1)[0]
        face_poses = face_poses[0]
        body_poses = body_poses[0]

        return face_out_pose, body_out_pose, face_poses, body_poses, face_clips, body_clips

    def predict_step(self, batch, batch_idx):
        if self.last_aud == batch['aud_file'][0]:
            return

        face_out_pose, body_out_pose, face_poses, body_poses, face_clips, body_clips = self.inference(batch, batch_idx)

        self.last_aud = batch['aud_file'][0]
        if not self.do_eval:
            self.renderer.render(self.method_name, face_out_pose, body_out_pose,
                                 face_poses, body_poses,
                                 batch['aud_file'][0], batch['betas'][0])

        else:
            pred = torch.cat([face_out_pose[:, :3], body_out_pose, face_out_pose[:, -100:]], dim=-1)
            gt = torch.cat([face_poses[:, :3], body_poses, face_poses[:, -100:]], dim=-1)

            return {
                "pose": pred.cpu(),
                "target": gt.cpu(),
                "betas": batch['betas'][0].cpu(),
                "audio": batch['aud_file'][0]
            }

    def on_predict_start(self):
        self.last_aud = None
        self.renderer = Renderer()

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        pass

    def validation_step(self, batch, batch_idx):
        self.renderer = Renderer()
        face_out_pose, body_out_pose, face_poses, body_poses, _, _ = self.inference(batch, batch_idx)

        self.renderer.render(f'{self.method_name}-val', face_out_pose, body_out_pose,
                             face_poses, body_poses,
                             batch['aud_file'][0], batch['betas'][0],
                             overwrite=True)

        self.renderer = None