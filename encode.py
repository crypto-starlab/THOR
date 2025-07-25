import pickle

import sys
import os 
project_root = os.path.abspath(os.path.join(os.getcwd(), './src'))
if project_root not in sys.path:
    sys.path.append(project_root)

from thor import ThorModelEncoder, CkksEngine

# Generate ckks engine
params = {"logN":16, "scale_bits": 41, "num_special_primes": 4, "devices": [0], "quantum":"pre_quantum"}

engine = CkksEngine(params)

#Encode model
for dataset_type in ['mrpc']:
    model_dir = f"./finetuned_models/{dataset_type}/model.safetensors"
    encoder = ThorModelEncoder(engine, model_dir)
    print("-" * 50)
    print(f"Encoding start for {dataset_type}")
    encoder.encode_pooler()
    with open(f"encoded_models_new/{dataset_type}/pooler.pkl", 'wb') as f:
        pickle.dump(encoder.weights_pt, f)
    print(f"Encoding complete for {dataset_type} Pooler")
    for layer in range(12):
        print(layer)
        encoder.encode_ff(layer)
    with open(f"encoded_models_new/{dataset_type}/ff.pkl", 'wb') as f:
        pickle.dump(encoder.weights_pt, f)
    print(f"Encoding complete for {dataset_type} FF")
    print("-" * 50)
    encoder = ThorModelEncoder(engine, model_dir)
    print(f"Encoding start for {dataset_type} Attention")
    for layer in range(12):
        print(layer)
        encoder.encode_att(layer)
    with open(f"encoded_models_new/{dataset_type}/att.pkl", 'wb') as f: ####
        pickle.dump(encoder.weights_pt, f) ####
    encoder.encode_cls()
    with open(f"encoded_models_new/{dataset_type}/cls.pkl", 'wb') as f:
        pickle.dump(encoder.weights_pt, f)
    print(f"Encoding complete for {dataset_type}")
    print("-" * 50)