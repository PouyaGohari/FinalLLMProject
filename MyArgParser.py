import argparse

def arg_parser():
    parser = argparse.ArgumentParser()
    ## Provide your token
    parser.add_argument("--hf_token", required=True,  type=str)
    ## For applying arrow or gks.
    parser.add_argument("--top_k", default=3, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--gks", action='store_true', help="Use general knowledge subtraction")
    parser.add_argument("--base_model_name", type=str, default="microsoft/Phi-3-mini-4k-instruct", required=True)
    parser.add_argument("--n_samples", type=int, default=64)
    ## For downloading general dataset like wiki dataset.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_path", default=None, type=str)
    parser.add_argument("--batch", default=8, type=int)
    ## For applying cka
    parser.add_argument("--export_data", action="store_true")
    parser.add_argument("--show_plot", action="store_true")
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--save_path", type=str)
    return parser.parse_args()
