import numpy as np
import soundfile as sf
from espnet2.bin.spk_inference import Speech2Embedding

import torch
import glob
import pickle as pk
import torch.nn.functional as F
from tqdm import tqdm

def adjust_audio_length(audio, sample_rate, length):
    current_length = len(audio)
    target_length = sample_rate * length # 10s

    if current_length > target_length:
        adjusted_audio = audio[:target_length]
    elif current_length < target_length:
        repeat_times = target_length // current_length
        remainder = target_length % current_length
        adjusted_audio = np.tile(audio, repeat_times)
        adjusted_audio = np.concatenate((adjusted_audio, audio[:remainder]))
    else:
        adjusted_audio = audio
    return adjusted_audio

if torch.cuda.is_available():
    print("Cuda available, conducting inference of GPU")
    gpu = True
else:
    print("Cuda not available, conducting inference of CPU")
    gpu = False

d_utt = {}

d_utt["SOMOS"] = glob.glob("/data/user_data/hyejinsh/corpus/SOMOS/audios/*.wav")
d_utt["NISQA_TEST"] = glob.glob("/data/user_data/hyejinsh/corpus/NISQA_Corpus/NISQA_TEST_*/*/*.wav")
d_utt["NISQA_VAL"] = glob.glob("/data/user_data/hyejinsh/corpus/NISQA_Corpus/NISQA_VAL_*/*/*.wav")
d_utt["VoiceMOS"] = glob.glob("/data/user_data/hyejinsh/corpus/VoiceMOS/DATA/wav/*.wav")
d_utt["Libri-train-clean-100"] = glob.glob("/data/group_data/swl/corpora/LibriSpeech/LibriSpeech/train-clean-100/*/*/*.flac")
d_utt["DAPS"] = glob.glob("/data/user_data/hyejinsh/corpus/daps/clean/*.wav")
d_utt["TSP"] = glob.glob("/data/user_data/hyejinsh/corpus/TSP_speech_48k/*/*.wav")

exp_list = {"CA", "CB", "FA", "FG", "ML", "MK"}
for utt in d_utt["TSP"]:
    if utt.split("/")[-2] in exp_list:
        d_utt["TSP"].remove(utt)
print (len(d_utt["TSP"]))

speech2spk_embed = Speech2Embedding.from_pretrained(model_tag="espnet/voxcelebs12_ska_wavlm_joint", device="cuda")

for key, utt_list in tqdm(d_utt.items()):
    embed_dic = {}
    for test_utt in tqdm(utt_list):
        #audio, sample_rate = sf.read(test_utt, frames=960000) # 1m
        audio, sample_rate = sf.read(test_utt) 
        audio = adjust_audio_length(audio, sample_rate, 10) # 10s

        embedding = speech2spk_embed(audio).detach().cpu().numpy().flatten()
        embed_dic[test_utt] = embedding

    with open("{}_spkembed_ska_wavlm_joint_10s".format(key), "wb") as f:
        pk.dump(embed_dic, f)


