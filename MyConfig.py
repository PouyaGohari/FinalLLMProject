MODEL_NAME='microsoft/Phi-3-mini-4k-instruct'
EXPERT_REPO_ID = "AliEdalat/le_experts_phi3_diff_lang"
# We are using colab you should change this cluster names with any directory you desire.
CLUSTER_NAMES = {f'cluster{i}':f"TahaBa/phi3-mini-clustered-flan/ts_expert_{i}" for i in range(10)}
# Since we are using a single expert in English in our directory is a single dictionary. Change it with your configurations.
LANGUAGE_EXPERTS = {'English_adapter': 'TahaBa/phi3-mini-general-adapters/cluster0_batch16_prop1.0_langen/checkpoint-16'}
DATA_FILE = {'train': 'en_Wiki_10k_LM_511_1 (1).json', 'test': 'en_Wiki_10k_LM_511_1_test (1).json'}
TARGET_MODULES = ["qkv_proj", "o_proj"]
MAX_LENGTH=1024