import regex as re
from .helper import get_stats,merge
from pathlib import Path


# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

#FILE_DIR = Path("/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/src/data/input_data/data/token_training/")
FILE_DIR = Path("/content/AI/LLM_research_platform/src/data/input_data/data/token_training/")

# for tokenization:
# 0: for padding
# 1: for EOS
# tokenization starts from 2
PAD_ID = 0
EOS_ID = 1 
TOKEN_OFFSET = 2
class RegexTokenizer:
    def __init__(self, file_dir = FILE_DIR, pattern = None, vocab_size = 5000): 
        self.file_dir = Path(file_dir)
        # pattern
        self.pattern = GPT2_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.vocab_size = vocab_size
        self.merge_dict = {}
        self.vocab_dict = {}
        
    def get_vocab_size(self):
        return self.vocab_size
    
    def train(self): #this function should be called right after instantiating RegexTokenizer
        chunks = []
        for file_path in self.file_dir.glob("*.txt"):
            print(f"training on {file_path}")
            with open(file_path, "r", encoding = "utf-8") as f:
                text = f.read()
            chunks.extend(self.compiled_pattern.findall(text))
        chunk_ids = [[x + TOKEN_OFFSET for x in ch.encode("utf-8")]for ch in chunks]

        merge_dict = {} # for encode {int,int} -> int
        # 0 = PAD
        # 1 = EOS
        # 2-257 = 256 byte tokens
        vocab_dict = {idx + TOKEN_OFFSET: bytes([idx]) for idx in range(256)} # for decode int -> bytes_object
        merge_rounds = self.vocab_size - TOKEN_OFFSET - 256
        for i in range(merge_rounds):
            counts = {}
            for ids in chunk_ids:
                get_stats(ids,counts)
            # Small training corpus may run out of pairs
            if not counts:
                break
            pair = max(counts, key = counts.get)
            idx = 256 + TOKEN_OFFSET + i
            chunk_ids = [merge(ids, pair, idx) for ids in chunk_ids]

            # save merge
            merge_dict[pair] = idx
            vocab_dict[idx] = vocab_dict[pair[0]] + vocab_dict[pair[1]]
            print(f"merged {pair[0]} and {pair[1]} into {idx}")
        
        self.merge_dict = merge_dict
        self.vocab_dict = vocab_dict

    def _encode_chunk(self, chunk_ids): #chunk_ids is list of decimal representations of bytes from the entire text after applying formate seperation
        while len(chunk_ids) >= 2:
            counts = {}
            get_stats(chunk_ids,counts)
            if not counts:
                break
            pair = min(counts, key = lambda p: self.merge_dict.get(p, float("inf")))
            if pair not in self.merge_dict:
                break
            idx = self.merge_dict[pair]
            chunk_ids = merge(chunk_ids, pair, idx)
        return chunk_ids

    def encode(self, text):
        chunks_text = self.compiled_pattern.findall(text)
        encoded_ids = []

        for chunk in chunks_text:
            chunk_ids = [x + TOKEN_OFFSET for x in chunk.encode("utf-8")]
            encoded_ids.append(self._encode_chunk(chunk_ids))

        # deal with EOS
        if encoded_ids:
            encoded_ids[-1].append(EOS_ID)
        else:
            encoded_ids.append([EOS_ID])

        return encoded_ids

    def decode(self, ids): #ids is a list of integers, return a python string
        bytes_list = []
        for idx in ids:
            if idx == EOS_ID:
                break
            if idx == PAD_ID:
                continue
            if idx in self.vocab_dict:
                bytes_list.append(self.vocab_dict[idx])
            else:
                raise ValueError(f"invalid token id: {idx}")
        bytes_text = b"".join(bytes_list)
        text = bytes_text.decode("utf-8", errors="replace")
        return text