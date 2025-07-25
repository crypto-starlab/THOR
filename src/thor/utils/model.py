from safetensors.torch import load_file as load_safetensors
from transformers import BertForNextSentencePrediction, AutoModelForSequenceClassification

def load_model(data_type:str, model_dir:str, type:str='default'):
    state_dict = load_safetensors(model_dir)
    if data_type in {'rte', 'mrpc'}:
        model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased', output_hidden_states=True)
    elif data_type == 'sst2':
        model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', output_hidden_states=True)
    elif data_type == 'stsb':
        model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1, output_hidden_states=True)
    model.load_state_dict(state_dict)
    print(f"Model loaded for {data_type}")
    return model

def inspect_model_weights(model_dir: str):
    state_dict = load_safetensors(model_dir)
    print("State dict keys and shapes:")
    for key, value in state_dict.items():
        print(f"{key}: {value.shape}")
        