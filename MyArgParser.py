import argparse
def downloading_adapters():
    parser = argparse.ArgumentParser(description="Download expert adapters from Hugging Face.")
    parser.add_argument("--hf_token", type=str)