import os.path
import shutil
import pickle
import random
import math
import json
from tqdm import tqdm
import lmdb
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import torchaudio as ta
import numpy as np
from transformers import Wav2Vec2Processor, HubertModel

from data_utils.show_data_utils.consts import speaker_id

with open('data_utils/show_data_utils/hand_component.json') as file_obj:
    comp = json.load(file_obj)
    left_hand_c = np.asarray(comp['left'])
    right_hand_c = np.asarray(comp['right'])

def to3d(data):
    left_hand_pose = np.einsum('bi,ij->bj', data[:, 75:87], left_hand_c[:12, :])
    right_hand_pose = np.einsum('bi,ij->bj', data[:, 87:99], right_hand_c[:12, :])
    data = np.concatenate((data[:, :75], left_hand_pose, right_hand_pose), axis=-1)
    return data

def linear_interpolation(features, input_fps, output_fps, output_len=None):
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = F.interpolate(features, size=output_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)



class SHOWDataset(Dataset):
    def __init__(self, data_root, cache_path, split, speakers, limbscaling=False, normalization=False,
                 norm_method="all", split_trans_zero=False, num_pre_frames=0, num_frames=88,
                 num_generate_length=88, aud_feat_win_size=None, aud_feat_dim=64, feat_method="mfcc",
                 context_info=False, smplx=True, audio_sr=16000, convert_to_6d=False, expression=True):
        self.data_root = data_root
        self.split = split
        self.speakers = speakers
        self.audio_sr = audio_sr
        self.audio_feat_dim = aud_feat_dim
        self.audio_feat_win_size = aud_feat_win_size
        self.num_generate_length = num_generate_length
        self.num_pre_frames = num_pre_frames
        self.convert_to_6d = convert_to_6d
        self.expression = expression
        self.context_info = context_info
        self.feat_method = feat_method
        self.normalization = normalization
        self.norm_method = norm_method
        self.whole_video = False

        self.num_samples = 0

        self.hubert = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").to('cuda')
        self.hubert.eval()

        self.am = Wav2Vec2Processor.from_pretrained("vitouphy/wav2vec2-xls-r-300m-phoneme")
        self.am_sr = 16000

        cache_dir = f'{data_root}/{cache_path}/{split}'

        self.build_cache(cache_dir, False)

        self.lmdb_env = lmdb.open(cache_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.num_samples = txn.stat()["entries"]


    def build_cache(self, dir, overwrite=False):
        if os.path.exists(dir):
            if overwrite:
                shutil.rmtree(dir)
            else:
                return

        if not os.path.exists(dir):
            os.makedirs(dir)


        lmdb_env = lmdb.open(dir, map_size=1024 ** 3 * 1500)
        # with lmdb_env.begin() as txn:
        #     length = txn.stat()['entries']

        huaide = 0
        for speaker_name in self.speakers:
            speaker_root = os.path.join(self.data_root, speaker_name)
            videos = [v for v in os.listdir(speaker_root)]
            for vid in tqdm(videos, desc="Processing training data of {}......".format(speaker_name)):
                source_vid = vid
                vid_pth = os.path.join(speaker_root, source_vid, self.split)

                try:
                    seqs = [s for s in os.listdir(vid_pth)]
                except:
                    continue

                for s in seqs:
                    # key = "{:005}".format(self.num_samples).encode("ascii")

                    seq_root = os.path.join(vid_pth, s)
                    key = seq_root # correspond to clip******
                    audio_fname = os.path.join(speaker_root, source_vid, self.split, s, '%s.wav' % (s))
                    motion_fname = os.path.join(speaker_root, source_vid, self.split, s, '%s.pkl' % (s))
                    if not os.path.isfile(audio_fname) or not os.path.isfile(motion_fname):
                        huaide = huaide + 1
                        continue

                    # data stuff here
                    loaded_data = {}

                    f = open(motion_fname, 'rb+')
                    data = pickle.load(f)

                    betas = np.array(data['betas'])
                    jaw_pose = np.array(data['jaw_pose'])
                    leye_pose = np.array(data['leye_pose'])
                    reye_pose = np.array(data['reye_pose'])
                    global_orient = np.array(data['global_orient']).squeeze()
                    body_pose = np.array(data['body_pose_axis'])
                    left_hand_pose = np.array(data['left_hand_pose'])
                    right_hand_pose = np.array(data['right_hand_pose'])
                    full_body = np.concatenate(
                        (jaw_pose, leye_pose, reye_pose, global_orient, body_pose, left_hand_pose, right_hand_pose), axis=1)
                    assert full_body.shape[1] == 99

                    if self.convert_to_6d:
                        raise NotImplementedError
                    else:
                        full_body = to3d(full_body)
                        expression = np.array(data['expression'])
                        full_body = np.concatenate((full_body, expression), axis=1)

                    complete_data = full_body
                    complete_data = np.array(complete_data)

                    audio, sr_0 = ta.load(audio_fname)

                    if self.audio_sr != sr_0:
                        audio = ta.transforms.Resample(sr_0, self.audio_sr)(audio)
                    if audio.shape[0] > 1:
                        audio = torch.mean(audio, dim=0, keepdim=True)

                    hubert_features = self.hubert(audio.to(self.hubert.device)).last_hidden_state.cpu()
                    hubert_features = linear_interpolation(hubert_features, 50, 30, output_len=len(complete_data))

                    # split the sequence
                    num_generate_length = self.num_generate_length
                    num_pre_frames = self.num_pre_frames
                    seq_len = num_generate_length + num_pre_frames

                    index_list = list(range(0,
                                            complete_data.shape[0] - self.num_generate_length - self.num_pre_frames,
                                            6))
                    if self.split in ['test'] or self.whole_video:
                        index_list = list([0])

                    for index in index_list:
                        index_new = index + random.randrange(0, 5, 3)
                        if index_new + seq_len > complete_data.shape[0]:
                            index_new = index
                        index = index_new

                        if self.split in ['val', 'pre', 'test'] or self.whole_video:
                            index = 0
                            seq_len = complete_data.shape[0]
                        seq_data = []
                        assert index + seq_len <= complete_data.shape[0]

                        seq_data = complete_data[index:(index + seq_len), :]
                        seq_data = np.array(seq_data)

                        '''
                        audio feature，
                        '''

                        '''
                        raw audio
                        '''
                        if not self.context_info:
                            if not self.whole_video:
                                fps = 30
                                aindex = (index / fps) * self.audio_sr
                                aseq_len = (seq_len / fps) * self.audio_sr
                                anum_pre_frames = (num_pre_frames / fps) * self.audio_sr
                                raw_audio = audio[..., math.floor(aindex):math.ceil(math.floor(aindex) + aseq_len + anum_pre_frames)]
                            else:
                                raw_audio = audio

                        '''
                        hubert features
                        '''
                        if not self.context_info:
                            if not self.whole_video:
                                hubert_feat = hubert_features[:, index:(index + seq_len), :]
                            else:
                                hubert_feat = hubert_features

                        with lmdb_env.begin(write=True) as txn:
                            key = "{:005}".format(self.num_samples).encode("ascii")
                            value = {
                                'poses': seq_data[:, :165].astype(np.float64).transpose(1, 0),
                                'expression': seq_data[:, 165:].astype(np.float64).transpose(1, 0),
                                'speaker': speaker_id[speaker_name],
                                'aud_file': audio_fname,
                                'betas': betas,
                                'raw_aud': raw_audio,
                                'hubert': hubert_feat
                            }
                            value = pickle.dumps(value)
                            txn.put(key, value)
                            self.num_samples += 1

        print("num_samples:", self.num_samples)

    def get_mfcc(self, audio):
        t = ta.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=64,
            melkwargs={
                "n_fft": 2048,
                "hop_length": 1072,#536,
                "n_mels": 256,
                "mel_scale": "htk"
            }
        )
        return t(torch.from_numpy(audio))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        with self.lmdb_env.begin(write=False) as txn:
            key = "{:005}".format(index).encode("ascii")
            sample = txn.get(key)
            sample = pickle.loads(sample)

            sample['hubert'] = sample['hubert'].detach().numpy()
            sample['raw_aud'] = ta.transforms.Resample(16000, 32000)(sample['raw_aud'])
            sample['raw_aud'] = sample['raw_aud'].detach().numpy()

            return sample