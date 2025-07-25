from transformers import AutoTokenizer, DataCollatorWithPadding
import datasets
import torch
from torch.utils.data import DataLoader
import numpy as np
from liberate.fhe.data_struct import DataStruct

from .ckks import CkksEngine
from .utils.matrix import ld_entry

class ThorDataEncryptor:
    def __init__(self, dataset_type: str,  dataset:str="", 
                 embedding_model:torch.nn.Module=None, ckks_engine: CkksEngine=None, test=False):
        self.checkpoint = "bert-base-uncased" 
        
        dataset_type = dataset_type.lower()
        if dataset_type not in {'mrpc', 'rte', 'sst2', 'stsb'}:
            raise ValueError("Invalid dataset type")
        self.dataset_type = dataset_type
        
        if dataset == "":
            print("Downloading dataset")
            dataset = datasets.load_dataset("glue", self.dataset_type)
        else:
            print("Loading dataset from disk")
            dataset = datasets.load_from_disk(dataset)
            
        self.dataset = dataset
        del self.dataset['train']
        del self.dataset['test']
        if test:
            self.dataset['validation'] = self.dataset['validation'].select(range(1))

        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, model_max_length = 128)
        if self.dataset_type == 'stsb':
            for key in self.dataset.keys():
                self.dataset[key] = self.dataset[key].map(lambda x: {**x, 'label': x['label'] / 5})
        self.embedding_model = embedding_model
        self.engine = ckks_engine 
        self.tokenized_dataset = self.tokenize()
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.eval_dataloader = DataLoader(
            self.tokenized_dataset["validation"], batch_size=1, collate_fn=data_collator
        )       
        
    def encrypt_embedding(self, embedding:np.ndarray, pk, level:int=0) -> np.ndarray[DataStruct]: 
        """
        Return an array of size (4,) which contains 4 ciphertexts. 
        """
        if embedding.shape != (128, 768):
            raise ValueError("Shape of embedding should be (128, 768)")
        if self.engine is None:
            raise ValueError("CKKS engine is not provided")
        x_T = np.transpose(embedding) #(768, 128)
        x_blocks = np.vsplit(x_T, 6) 
        ct = np.empty((4,), dtype=object)
        for i in range(4): #16 * i + j = n =  0, 1, ..., 63 
            msg = np.zeros((2**15,), dtype=complex)
            for j in range(16): #l_0, l_1, ..., l_15 (mod 16)
                temp = j *(2**11)
                l = i * 16 + j
                for t in range(128): #d_0, d_1, ..., d_127
                    for b in range(12): #x_0, x_1, ..., x_5
                        x_b = x_blocks[b % 6]
                        msg[temp + t*16 + b] = complex(ld_entry(x_b, l, t), ld_entry(x_b, l+64, t))
            ct[i] = self.engine.encode_and_encrypt(msg, pk, level)
        return ct

    def encode_attention_mask(self, attention_mask:np.ndarray, level:int=15) -> np.ndarray[DataStruct]:
        """
        Return an array of size (8,) which contains 8 plaintexts. 
        """
        if attention_mask.shape != (128,):
            raise ValueError("Shape of attention mask should be (128,)")
        n_tokens = np.count_nonzero(attention_mask)
        attention_mask = np.full((8,), None, dtype=object)
        for i in range(8):
            msg = np.zeros((2**15,), dtype=float)
            for j in range(16):
                temp = j *(2**11)
                diag_index = i * 16 + j
                for t in range(128):
                    col_index = (diag_index + t) % 128
                    is_token = 1 if col_index < n_tokens else 0
                    for head in range(12):
                        msg[temp + t*16 + head] = is_token
            attention_mask[i] = self.engine.encode(msg, level)
        return attention_mask
    
    def encrypt_attention_mask(self, attention_mask:np.ndarray, level:int=0) -> np.ndarray[DataStruct]:  
        """
        Return an array of size (8,) which contains 8 ciphertexts. 
        """
        if attention_mask.shape != (128,):
            raise ValueError("Shape of attention mask should be (128,)")
        n_tokens = np.count_nonzero(attention_mask)
        attention_mask = np.full((8,), None, dtype=object)
        for i in range(8):
            msg = np.zeros((2**15,), dtype=float)
            for j in range(16):
                diag_index = i * 16 + j
                temp = j *(2**11)
                for t in range(128):
                    col_index = diag_index + t
                    is_token = 1 if col_index < n_tokens else 0
                    for head in range(12):
                        msg[temp + t*16 + head] = is_token
            attention_mask[i] = self.engine.encodecrypt(msg, level)
        return attention_mask    
                
    def embed_all_data(self) -> list[np.ndarray]:
        eval_dataloader = self.get_eval_dataloader()
        embeddings = []
        attention_masks = []
        labels = []
        for batch in eval_dataloader:
            inputs = {k: v for k, v in batch.items() if k in ['input_ids', 'token_type_ids']}
            with torch.no_grad():
                embedding = self.embedding_model(**inputs)
            embeddings.append(embedding.numpy().squeeze())
            attention_masks.append(batch['attention_mask'].numpy())
            labels.append(batch['labels'].numpy()[0])
            
        if len(embeddings) != len(attention_masks) or len(embeddings) != len(labels) or len(attention_masks) != len(labels):
            raise ValueError("Length of embeddings, attention_masks and labels should be equal")
        
        return embeddings

    def embed_data(self, input_data:dict) -> np.ndarray:
        with torch.no_grad():
            embedding = self.embedding_model(**input_data)
        return embedding.numpy().squeeze()
    
    def get_eval_dataloader(self, batch_size:int=1) -> DataLoader:
        return self.eval_dataloader
        
    def tokenize(self) -> datasets.DatasetDict:
        tokenized_dataset = self.dataset.map(self._tokenize, batched=True)
        if self.dataset_type in {'mrpc', 'rte', 'stsb'}:
            tokenized_dataset = tokenized_dataset.remove_columns(['sentence1', 'sentence2', 'idx']).rename_column('label', 'labels')
        elif self.dataset_type == 'sst2':
            tokenized_dataset = tokenized_dataset.remove_columns(['sentence', 'idx']).rename_column('label', 'labels')
        tokenized_dataset.set_format(type='torch')
        return tokenized_dataset
    
    def _tokenize(self, data):
        if self.dataset_type in {'mrpc', 'rte', 'stsb'}:
            return self.tokenizer(data['sentence1'], data['sentence2'],max_length=128,padding='max_length',truncation=True)
        elif self.dataset_type == 'sst2':
            return self.tokenizer(data['sentence'],padding='max_length',max_length=128,truncation=True)