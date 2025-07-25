import argparse

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", required=True,  type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_path", default=None, type=str)
    parser.add_argument("--top_k", default=3, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--gks", default=False, type=bool)
    parser.add_argument("--base_model_name", type=str, default="microsoft/Phi-3-mini-4k-instruct", required=True)
    parser.add_argument("--n_samples", type=int, default=64)
    return parser.parse_args()
