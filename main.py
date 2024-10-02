import numpy as np
import pickle as pk
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import argparse

from evaluation import * 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_embeddings(embed_dir, feature_set):
    """
    Load embeddings from the given directory and return them in a dictionary.
    """
    d_ref_feats = {}
    d_test_feats = {}

    with open(embed_dir + f"Libri-train-clean-100_wavlm_large_10s", "rb") as f:
        d_ref_feats["lm"] = pk.load(f)
    with open(embed_dir + f"Libri-train-clean-100_spkembed_ska_wavlm_joint_10s", "rb") as f:
        d_ref_feats["spk"] = pk.load(f)

    with open(embed_dir + f"VoiceMOS_wavlm_large_10s", "rb") as f:
        d_test_feats["lm"] = pk.load(f)
    with open(embed_dir + f"VoiceMOS_spkembed_ska_wavlm_joint_10s", "rb") as f:
        d_test_feats["spk"] = pk.load(f)

    return d_ref_feats, d_test_feats

def compute_similarities(l_eval_lines, d_ref_feats, d_test_feats, data_dir, k, feat, output_file):
    """
    Compute cosine similarities and write the top k most similar references to a file.
    """
    d_mos = {}
    d_topk_ref = {}

    with open(output_file, "w") as f_out:
        for line in tqdm(l_eval_lines):
            utt_eval, mos_score = line.strip().split(",")
            utt_eval_path = data_dir + utt_eval
            
            test_embed = d_test_feats[feat][utt_eval_path].reshape(1, -1)

            similarities = {}
            if utt_eval_path in d_test_feats[feat]:
                d_mos[utt_eval_path] = float(mos_score)
                for ref_file, ref_embed in d_ref_feats[feat].items():
                    similarity = cosine_similarity(test_embed, ref_embed.reshape(1, -1))
                    similarities[ref_file] = similarity

            # Get top k most similar references
            topk = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]

            refined_topk = [ref[0] for ref in topk]
            if utt_eval_path not in d_topk_ref:
                d_topk_ref[utt_eval_path] = refined_topk

            ref_line = ','.join(refined_topk)
            f_out.write(f"{utt_eval} {ref_line}\n")
    
    return d_mos, d_topk_ref

def compute_scores(d_topk_ref, d_test_feats, d_ref_feats, d_mos):
    """
    Compute SB scores and MOS correlations.
    """
    SB_list = []
    MOS_list = []

    for test_utt, refs in tqdm(d_topk_ref.items()):
        l_scores = []
        MOS_list.append(d_mos[test_utt])

        test_feat = convert_and_concatenate(d_test_feats["lm"], d_test_feats["spk"], test_utt)
        for ref in refs:
            ref_feat = convert_and_concatenate(d_ref_feats["lm"], d_ref_feats["spk"], ref)
            l_scores.append(calculate_bert(test_feat, ref_feat))

        SB_list.append(np.mean(l_scores))

    final_corr_scores = calculate_correlations(SB_list, MOS_list)
    return final_corr_scores

def main(args):
    # Load the evaluation list
    with open(args.eval_list, "r") as l_eval_file:
        l_eval_lines = l_eval_file.readlines()[1:]  # Skip header

    # Load reference and test embeddings
    d_ref_feats, d_test_feats = load_embeddings(args.embed_dir, ["lm", "spk"])

    # Compute similarities for each feature set (lm and spk)
    for feat in ["lm", "spk"]:
        output_file = f"./matching_reference/{args.test_db}_{args.ref_db}_top{args.k}_{feat}.txt"
        d_mos, d_topk_ref = compute_similarities(l_eval_lines, d_ref_feats, d_test_feats, args.data_dir, args.k, feat, output_file)

        # Compute SB and MOS scores
        final_corr_scores = compute_scores(d_topk_ref, d_test_feats, d_ref_feats, d_mos)
        print(f"Final correlation scores for {feat}: {final_corr_scores}")

# Argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process embeddings and compute cosine similarity for voice MOS.")
    parser.add_argument("--eval_list", type=str, required=True, help="Path to evaluation list file (test_mos_list.txt)")
    parser.add_argument("--embed_dir", type=str, required=True, help="Directory containing embedding files")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing wav files for testing")
    parser.add_argument("--test_db", type=str, default="VoiceMOS", help="Name of the test database")
    parser.add_argument("--ref_db", type=str, default="Libri-train-clean-100", help="Name of the reference database")
    parser.add_argument("--k", type=int, default=10, help="Top-k most similar references to retrieve")

    args = parser.parse_args()
    main(args)
