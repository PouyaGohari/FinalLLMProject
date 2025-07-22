import argparse
def downloading_adapters():
    parser = argparse.ArgumentParser(description="Download expert adapters from Hugging Face.")
    parser.add_argument("--hf_token", required=True,  type=str)
    return parser.parse_args()