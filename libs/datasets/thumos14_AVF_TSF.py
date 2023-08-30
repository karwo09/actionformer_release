import os
import librosa
import random
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats
from numpy.lib.stride_tricks import as_strided

@register_dataset("thumos_AVF_TSF")
class THUMOS14CLIPDataset(Dataset):
    def __init__(
        self,
        is_training,     # if in training mode
        split,           # split, a tuple/list allowing concat of subsets
        feat_folder,     # folder for features
        audio_folder,    # folder for audio
        json_file,       # json file for annotations
        feat_stride,     # temporal stride of the feats
        num_frames,      # number of frames for each feat
        default_fps,     # default fps
        downsample_rate, # downsample rate for feats
        max_seq_len,     # maximum sequence length during training
        trunc_thresh,    # threshold for truncate an action segment
        crop_ratio,      # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,       # input feat dim
        num_classes,     # number of action categories
        file_prefix,     # feature file prefix if any
        file_ext,        # feature file extension if any
        force_upsampling # force to upsample to max_seq_len
    ):
        # file path
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        self.audio_folder = audio_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio
        
        # audio
        # self.SAMPLE_RATE = 16000 # 16kHz as per https://arxiv.org/abs/2106.14118
        
        #TODO: Need to check if AudioCLIP can consume less VRAM
        self.SAMPLE_RATE = 4198 # 2kHz to match the video frame rate
        self.audio_extention = '.csv'

        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)
        assert len(label_dict) == num_classes
        self.data_list = dict_db
        self.label_dict = label_dict

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'thumos-14-clip',
            'tiou_thresholds': np.linspace(0.3, 0.7, 5),
            # we will mask out cliff diving
            'empty_label_ids': [],
        }

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        # if label_dict is not available
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                for act in value['annotations']:
                    label_dict[act['label']] = act['label_id']

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue
            # or does not have the feature file
            feat_file = os.path.join(self.feat_folder,
                                     self.file_prefix + key + self.file_ext)
            if not os.path.exists(feat_file):
                continue

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."

            # get video duration if available
            if 'duration' in value:
                duration = value['duration']
            else:
                duration = 1e8

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                # a fun fact of THUMOS: cliffdiving (4) is a subset of diving (7)
                # our code can now handle this corner case
                segments, labels = [], []
                for act in value['annotations']:
                    segments.append(act['segment'])
                    labels.append([label_dict[act['label']]])

                segments = np.asarray(segments, dtype=np.float32)
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
            else:
                segments = None
                labels = None
            active_label = value['annotations'][0]['label']
            dict_db += ({'id': key,
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels,
                         'active_label' : label_dict[active_label]
            }, )
        self.label_dict = label_dict
        return dict_db, label_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]

        # load features
        filename = os.path.join(self.feat_folder,
                                self.file_prefix + video_item['id'] + self.file_ext)
        feats = np.load(filename).astype(np.float32)
        
        # Load audio:
        # Divide by current fps and multiply by original 30fps then the stride is 4 and padding is 2 on each side
        # (T_a/16*30/4)-4 gives the amount of feats in the video -4 indicates padding with zeros probably
        # This can be fixed with strides:
        # (T- T/2 - T/32)-4, stride of 2 and a stride of 32 and padding of 2 on each side
        if(self.audio_folder):
            audio_filename = os.path.join(self.audio_folder,
                                    self.file_prefix + video_item['id'] + self.audio_extention)
            track = np.genfromtxt(audio_filename, delimiter=",")
        else:
            # Using default audio folder
            audio_filename = os.path.join(self.feat_folder,
                                    self.file_prefix + video_item['id'] + self.audio_extention)
        track = np.genfromtxt(audio_filename, delimiter=",") # (T, 128)
        # track, _ = librosa.load(audio_filename, sr=self.SAMPLE_RATE, dtype=np.float32)

        # deal with downsampling (= increased feat stride)
        feats = feats[::self.downsample_rate, :]
        
        # this is the downsampling rate for the audio shown above
        # df1 = df [df.index % 3 != 0] 
        # Excludes every 3rd row starting from 0 df2 = df [df.index % 3 == 0] 
        # Selects every 3rd raw starting from 0.
        # Create a mask for the strides:
        T = track.shape[0]
        x = (T - T//2 - T//32) - 4
        new_shape = (track.shape[0]//x, x) + track.shape[1:]
        new_strides = (x*track.strides[0],) + track.strides
        track = as_strided(track, shape=new_shape, strides=new_strides)
        if len(track.shape) == 3:
            track = np.mean(track, axis=0)
        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = 0.5 * self.num_frames / feat_stride
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))
        # track = torch.from_numpy(np.ascontiguousarray(track.transpose()))
        track = torch.from_numpy(track)

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
            )
            labels = torch.from_numpy(video_item['labels'])
            
        else:
            segments, labels = None, None
        rand = random.randint(0,(len(video_item['labels'])-1))
        activity = str([i for i in self.label_dict if self.label_dict[i]==video_item['labels'][int(rand)]][0])
        if activity is None:
            activity = 'Number ' +str(video_item['labels'][int(rand)])
        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : video_item['fps'],
                     'prompt'          : activity,
                    #  'prompt'          : "A person doing " + activity + ".",
                     'audio_track'     : track,
                     'duration'        : video_item['duration'],
                     'active_label'    : video_item['labels'][int(rand)],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : self.num_frames}

        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
            )

        return data_dict
