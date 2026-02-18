import os
import pickle
import logging
import numpy as np
from itertools import islice
import torch
from torchvision import transforms

from .dataset import SkeletonDataset
from .augmentations import GaussianNoise, Reflect, Rotation



class MabeMouseDataset(SkeletonDataset):
    
    DEFAULT_GRID_SIZE = 850
    SAMPLE_LEN = 1800
    NUM_INDIVIDUALS = 3

    def __init__(self, 
                 path_to_data_dir, 
                 sampling_rate = 1, 
                 num_frames = 80, 
                 sliding_window = 100, 
                 if_fill = True,
                 # patch_size: tuple = (6, 1, 2), 
                 cache_path = None, 
                 cache = True, 
                 augmentations = True, #centeralign: bool = False, 
                 include_testdata: bool = False,
                 **kwargs):
        
        self.include_testdata = include_testdata
        
        if augmentations:
            gs = (self.DEFAULT_GRID_SIZE, self.DEFAULT_GRID_SIZE)
            self.augmentations = transforms.Compose(
                [Rotation(grid_size=gs, p=0.5),
                 GaussianNoise(p=0.5),
                 Reflect(grid_size=gs, p=0.5),]
            )
        else:
            self.augmentations = None

        super().__init__(path_to_data_dir, sampling_rate, num_frames, sliding_window, if_fill, #patch_size, 
                         cache_path, cache, **kwargs) # this calls Baseclass.__init__(self, ....)



    def load_data(self):
        """Load raw data"""
        self.raw_data = np.load(self.path_to_data_dir, allow_pickle=True).item()
        self.raw_data = dict(islice(list(self.raw_data.items()), 5))
        if self.include_testdata:
            raw_data_test = np.load(self.path_to_data_dir.replace("_train.npy", "_test.npy"), allow_pickle=True).item()
            self.raw_data["sequences"].update(raw_data_test["sequences"])


    def check_annotations(self) -> None:
        """Annotation check handler"""
        self.has_annotations = "vocabulary" in self.raw_data.keys()
        if self.has_annotations:
            self.annotation_names = self.raw_data["vocabulary"]


    @staticmethod
    def fill_holes(data):
        """Fill zero """
        clean_data = data.copy()
        num_frames, num_individuals, num_joints, _ = clean_data.shape
        # Fill frame 0 using future frames
        for m in range(num_individuals):
            holes = np.where(clean_data[0, m, :, 0] == 0)[0]
            for h in holes:
                valid = np.where(clean_data[:, m, h, 0] != 0)[0]
                if valid.size > 0:
                    clean_data[0, m, h, :] = clean_data[valid[0], m, h, :]
        # Forward-fill remaining frames
        for fr in range(1, num_frames):
            for m in range(num_individuals):
                holes = np.where(clean_data[fr, m, :, 0] == 0)[0]
                clean_data[fr, m, holes, :] = clean_data[fr - 1, m, holes, :]


    def preprocess(self):
        """Initial preprocessing"""
        self.check_annotations()

        sequences = self.raw_data["sequences"]
        seq_keypoints = []
        keypoints_ids = []
        sub_seq_length = self.max_keypoints_len
        # self.labels = {key: [] for key in self.annotation_names}
        
        for seq_ix, (seq_name, sequence) in enumerate(sequences.items()): #index ,(mouse_name, value)
            
            vec_seq = sequence["keypoints"] # one seqeunces (1800, 3, 12, 2)
            if self.if_fill:
                vec_seq = self.fill_holes(vec_seq)
            if self.sampling_rate > 1:
                vec_seq = vec_seq[:: self.sampling_rate]
            # Pads the beginning and end of the sequence with duplicate frames
            pad_vec = np.pad(vec_seq,
                             ((sub_seq_length// 2, sub_seq_length - 1 - sub_seq_length // 2), (0, 0), (0, 0), (0, 0)), mode="edge", )
            seq_keypoints.append(pad_vec)
            
            #for i in range(len(self.annotation_names)): # Store the labels for each subsequence (if annotations are available)
            #    self.labels[self.annotation_names[i]].append(sequence["annotations"][i])
            keypoints_ids.extend([(seq_ix, i) for i in np.arange(0, len(pad_vec) - sub_seq_length + 1, self.sliding_window)]) # (1600 * num_samples/sequences, T=600, M=3, V=12, C=2)
            
        self.seq_keypoints = np.array(seq_keypoints, dtype=np.float32) # (1600, 1800, C=3, 12, 2) -> 
        
        print("seq_keypoints shape: ", self.seq_keypoints.shape)
        
        self.keypoints_ids = keypoints_ids
        print("keypoints_ids length: ", len(self.keypoints_ids))
        # for label_name in self.annotation_names:
        #    self.labels[label_name] = np.array(self.labels[label_name], dtype=np.float32)

        del self.raw_data


    def save_processed_data(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "wb") as output:
            #pickle.dump({"keypoints": self.seq_keypoints, "labels": self.labels}, output)
            pickle.dump(self.seq_keypoints, output)
        logging.info("Processed data was saved to {}.".format(self.cache_path))


    def load_from_processed(self):
        logging.warning( f"Loading processed data from {self.cache_path}. "
                         "Delete this file or set cache=False if processing changed.")
        with open(self.cache_path, "rb") as fp:
            # self.seq_keypoints, self.labels = pickle.load(fp)
            self.seq_keypoints = pickle.load(fp)
   
    
    def normalize(self, data):
        """Scale by dimensions of image and mean-shift to center of image."""
        state_dim = data.shape[-1] // 2
        shift = np.array([self.DEFAULT_GRID_SIZE / 2, self.DEFAULT_GRID_SIZE / 2] * state_dim)
        scale = shift.copy()

        return (data - shift) / scale

    
    def prepare_subsequence_sample(self, sequence: np.ndarray): # sequence :(sample_length, 3, 12, 2)
        if self.augmentations:
            sequence = self.augmentations(sequence)
        #sequence = sequence.reshape(self.max_keypoints_len, -1)      # simply flatten
        
        keypoints = self.normalize(sequence) # sequnece should be in shape (sample_length, 3*12*2)
        #if self.centeralign:
        #    keypoints = keypoints.reshape(self.max_keypoints_len, *self.KEYFRAME_SHAPE)
        #    keypoints = self.transform_to_centeralign_components(keypoints)
        feats = torch.tensor(keypoints, dtype=torch.float32)#.unsqueeze(keypoints, 0)
        return feats

    def __len__(self):
        return len(self.keypoints_ids)  # sub sequence
    
    def __getitem__(self, idx: int):
        subseq_ix = self.keypoints_ids[idx]
        subsequence = self.seq_keypoints[subseq_ix[0], subseq_ix[1] : subseq_ix[1] + self.max_keypoints_len] # (length, 3, 12, 2)
        feats = self.prepare_subsequence_sample(subsequence)
        return feats, []
    
    
    """
    # Version 2: each sample is the whole sequence
    def prepare_sequence_sample(self, sequence: np.ndarray):
        if self.augmentations:
            sequence = self.augmentations(sequence)
        feats = torch.tensor(self.normalize(sequence, self.DEFAULT_GRID_SIZE), dtype=torch.float32) # (1800, 3, 12, 2)
        feats = feats.reshape(self.SAMPLE_LEN, self.NUM_INDIVIDUALS, -1) # (1800, 3, 24)
        feats = feats.permute(1, 0, 2) # (3, 1800, 24)
        return feats
    
    def __len__(self):
        return len(self.seq_keypoints) # whole sequence
    
    def __getitem__(self, idx: int):
        sequence = self.seq_keypoints[idx] # (1800, 3, 12, 2)
        feats = self.prepare_sequence_sample(sequence)
        return feats, []
    """