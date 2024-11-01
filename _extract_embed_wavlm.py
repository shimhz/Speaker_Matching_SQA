import os
import numpy as np
import soundfile as sf
import torchaudio
import torch
import torch.nn.functional as F
from transformers import WavLMModel
import pickle as pk
import glob
from tqdm import tqdm
import logging
import argparse

logging.getLogger("transformers").setLevel(logging.ERROR)

# Function to adjust the audio length to a target length
def adjust_audio_length(audio, sample_rate, length):
    current_length = len(audio)
    target_length = int(sample_rate * length)
    
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

# Function to process each dataset and save embeddings
def process_utterances(utt_list, model, layer, device, output_file, sample_rate=16000):
    embed_dic = {}
    for test_utt in tqdm(utt_list):
        audio, sr = sf.read(test_utt, frames=160000)  # Read 10 sec
        audio = adjust_audio_length(audio, sr, 10)
        audio = torch.from_numpy(audio).unsqueeze(0).to(device).float()

        if sr != sample_rate:
            audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(audio)

        # Extract features
        feats_hiddens = model(audio, output_hidden_states=True).hidden_states
        embedding = feats_hiddens[layer]

        # Store embedding
        embed_dic[test_utt] = embedding.detach().cpu().numpy()

    # Save embeddings to a file
    with open(output_file, "wb") as f:
        pk.dump(embed_dic, f)

    del embed_dic  # Clear memory

# Main function to run the inference and embedding extraction
def main(args):
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    # Load the model
    model = WavLMModel.from_pretrained("microsoft/wavlm-large")
    model.eval()
    model = model.to(device)
    layer = args.layer

    # Collect files for each dataset
    d_utt = {}
    d_utt["Libri-train-clean-100"] = glob.glob(os.path.join(args.data_dir, "LibriSpeech/train-clean-100/*/*/*.flac"))
    d_utt["VoiceMOS"] = glob.glob(os.path.join(args.data_dir, "VoiceMOS/DATA/wav/*.wav"))

    # Process and filter the TSP dataset (if applicable)
    if "TSP" in d_utt:
        exp_list = {"CA", "CB", "FA", "FG", "ML", "MK"}
        d_utt["TSP"] = [utt for utt in d_utt["TSP"] if utt.split("/")[-2] not in exp_list]
        print(f"TSP dataset filtered: {len(d_utt['TSP'])} remaining")

    # Process each dataset
    for key, utt_list in d_utt.items():
        print(f"Processing {key}")
        output_file = f"{key}_wavlm_large_10s"
        process_utterances(utt_list, model, layer, device, output_file)

# Argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract WavLM embeddings from audio files.")

    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing datasets.")
    parser.add_argument("--layer", type=int, default=14, help="WavLM layer to extract embeddings from.")
    
    args = parser.parse_args()
    
    main(args)
