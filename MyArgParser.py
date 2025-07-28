import argparse

from accelerate.commands.config.default import description


def arg_parser():
    parser = argparse.ArgumentParser()
    ## Provide your token
    parser.add_argument("--hf_token", required=True,  type=str, description="Your huggingface token.")
    ## For applying arrow or gks.
    parser.add_argument("--top_k", default=3, type=int, description="This is top_k of arrow.")
    parser.add_argument("--temperature", default=1.0, type=float, description="The temperature of arrow's softmax.")
    parser.add_argument("--gks", action='store_true', help="Use general knowledge subtraction")
    parser.add_argument("--base_model_name", type=str, default="microsoft/Phi-3-mini-4k-instruct", required=True, description="The base model.")
    parser.add_argument("--n_samples", type=int, default=64, description="The number of samples from the general dataset (e.g, Wiki dataset)")
    ## For downloading general dataset like wiki dataset.
    parser.add_argument("--seed", type=int, default=42, description="For reproducibility.")
    parser.add_argument("--dataset_path", default=None, type=str, description="The path of your dataset.")
    ## For applying cka
    parser.add_argument("--export_data", action="store_true", description="If you want to export data from comparing CKA of two models.")
    parser.add_argument("--show_plot", action="store_true", description="If you want to plot the CKA result.")
    parser.add_argument("--device", default='cuda', type=str, description="The device you want to apply CKA.")
    return parser.parse_args()
