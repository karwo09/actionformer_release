import argparse
import os, glob
from numpy import genfromtxt
from pydub import AudioSegment
import numpy as np
import torchaudio
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.client import device_lib
import torch
import json
print(device_lib.list_local_devices())

parser = argparse.ArgumentParser(description='Tool to extract audio feature sequence using VGGish')

parser.add_argument(
    '--input',
    default='ActivityNETaudio',
    help='provide the input directory containing audio files (default: ActivityNETaudio)'
)

parser.add_argument(
    '--output',
    default='VGGishFeatures',
    help='path to save VGGish features (default: VGGishFeatures)'
)

parser.add_argument(
    '--snippet_size',
    default=1.2,
    help='snippet size in seconds (default: 1.2)'
)

parser.add_argument(
    '--sampling_rate',
    default=16000,
    help='Sampling rate of audio files (default: 16000)'
)

parser.add_argument(
    '--feature_frame_size',
    default=16,
    help='Size of each segment in terms of no. of frames (default: 16)'
)

parser.add_argument(
    '--annotations',
    help='File containing annotations (default: None)'
)



# Returns feature sequence from audio 'filename'
def getFeature(_filename, input_dir, output_dir,frames,snippet_size, vggmodel, sampling_rate=16000):
    if(os.path.isfile(input_dir + '/' + _filename.split('.')[0] +".csv")):
        print("File already exists: " + _filename)
        return 1
    try:
        frameCnt = frames[_filename.split('.')[0]]
    except:
        print("File not found in annotations: " + _filename)
        return 0
    # Convert m4a to wav
    filename = _filename.split('.')[0]
    if(_filename.split('.')[1] == 'm4a'):
        # Convert m4a to wav if not already converted
        file = AudioSegment.from_file(input_dir + '/' + filename+".m4a", "m4a")
        file.export(filename + '.wav', format="wav")

    # Initialize Feature Vector
    featureVec = tf.Variable([[ 0 for i in range(128) ]], dtype='float32')

    # Load audio file as tensor
    audio, sr = torchaudio.load(input_dir + '/' + filename + '.wav')
    # Convert to mono
    audio = audio.mean(axis=0)
    # Resample to 16kHz
    audio = torchaudio.transforms.Resample(sr, sampling_rate)(audio.view(1,-1))[0]

    # Iterate over all snippets and extract feature vector
    pointerr = len(audio) // frameCnt
    frameSize = len(audio) // frameCnt
    for i in range(frameCnt):
        # Get audio segment b/w start_time and end_time
        chunk = audio[max(0, pointerr - (snippet_size // 2)):min(len(audio), pointerr + (snippet_size // 2))]
        if len(chunk) < snippet_size:
            chunk = torch.from_numpy(np.pad(chunk, pad_width=(0, snippet_size - len(chunk)), mode='constant', constant_values=0))
        # Extract feature vector sequence
        feature = vggmodel(chunk)
        # Combine vector sequences by taking mean to represent whole segment. (i.e. convert (Ax128) -> (1x128))
        if len(feature.shape) == 2:
            feature = tf.reduce_mean(feature, axis=0)
        # Concatenate to temporal feature vector sequence
        featureVec = tf.concat([featureVec, [feature]], 0)
        pointerr += frameSize

    # Removing first row with zeroes
    featureVec = featureVec[1:].numpy()

    if(_filename.split('.')[1] == 'm4a'):
        os.remove(filename + ".wav") 

    # Save as csv
    np.savetxt(output_dir + '/' + filename + ".csv", featureVec, delimiter=",", header=','.join([ 'f' + str(i) for i in range(featureVec.shape[1]) ]))
    print(filename + ' Done')
    return 1
    
def main(args):
    if args is None:
        my_namespace = parser.parse_args()
        input_dir = my_namespace.input
        output_dir = my_namespace.output
        audio_snippet_size = my_namespace.snippet_size
        sampling_rate = my_namespace.sampling_rate
        feature_frame_size = my_namespace.feature_frame_size
        annotations = my_namespace.annotations
    else:
        input_dir = args.input
        output_dir = args.output
        audio_snippet_size = args.snippet_size
        sampling_rate = args.sampling_rate
        feature_frame_size = args.feature_frame_size
        annotations = args.annotations

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # VGGish feature extractor model
    vggmodel = hub.load('https://tfhub.dev/google/vggish/1')

    # Snippet size (In terms of no. of frames).
    audio_snippet_size = int(16000 * audio_snippet_size)

    # This file contains no. of frames for each video.
    # This is used to extract feature sequence of same length for all videos.
    # Depreciated
    # data = np.genfromtxt('video_info.csv', delimiter=',', dtype=str)
    # frames = {}
    # for i in data[1:]:
    #     frames[i[0]] = int(i[1])
    
    # Read all files
    # fileNames = sum([ glob.glob(input_dir+"/"+x) for x in ("*.npy") ], [])
    
    fileNames = glob.glob(input_dir+"/*.npy")
    fileNames = [ os.path.basename(file) for file in fileNames ]
    # fileNames.sort()
    # fileNames = fileNames[int(len(fileNames) / 2):]
    print(str(len(fileNames)) + " files found...")
    
    with open(annotations) as f:
        d = json.load(f)
        data = d['database']
    if (data == None):
        raise Exception('No data found in annotations file')
    frames = {}
    for i in data:
        frames[i] = int(data[i]['duration'] * feature_frame_size) # number of seconds * frames per second = tot number of frames
    


    count = 0
    # Extract temporal feature sequence for all audio files.
    for file in fileNames:
        print("Extracting features for " + file)
        count += getFeature(file, input_dir, output_dir, frames, audio_snippet_size, vggmodel, sampling_rate=sampling_rate)
        print(str(count) + "/" + str(len(frames)) + " files processed successfully.")

if __name__ == '__main__':
    # main(None)
    class Args:
        def __init__(self):
            self.input = 'data/thumos/i3d_features'
            self.output = 'data/thumos/i3d_features'
            self.snippet_size = 1.2
            self.sampling_rate = 16000
            self.feature_frame_size = 16
            self.annotations = "data/thumos/annotations/thumos14.json"
        
    args = Args()
    main(args)