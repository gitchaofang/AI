import regex as re
from .helper import get_stats
from .helper import merge

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer:
    def __init__(self,filename, pattern = None, vocab_size = 255):
        #path = "/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/src/data/input_data/data/" + file_name # for local
        path = "/content/AI/LLM_research_platform/src/data/input_data/data/" + filename # for codlab online # for codelab online
        with open(path, "r", encoding="utf-8") as f:
            self.text = f.read() 

        # pattern
        self.pattern = GPT2_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.vocab_size = vocab_size
        self.merge_dict = {}
        self.vocab_dict = {}
        

    def train(self, pattern = None):
        chunks = re.findall(pattern, self.text)
        chunk_ids = [list(ch.encode("utfg-8")) for ch in chunks]

        merge_dict = {} # for encode {int,int} -> int
        vocab_dict = {idx: bytes([idx]) for idx in range(255)} # for decode int -> bytes_object
        merge_rounds = self.vocab_size - 255
        for i in range(merge_rounds):
            counts = {}
            for ids in chunk_ids:
                get_stats(ids,counts)
            pair = max(counts, key = counts.get)
            idx = 255 + i
            chunk_ids = [merge(ids, pair, idx) for ids in chunk_ids]

            # save merge
            merge_dict[pair] = idx
            vocab_dict[idx] = vocab_dict[pair[0]] + vocab_dict[pair[1]]
            print(f"merged {pair[0]} and {pair[1]} into idx")
        
        self.merge_dict = merge_dict
        self.vocab_dict = vocab_dict

    def _encode_chunk(self, chunk_ids): #chunk is each decimal representations (list) of bytes from the entire text after applying formate seperation
        while len(chunk_ids) >= 2:
            counts = get_stats(chunk_ids)
            pair = min(counts, key = lambda p: self.merge_dict.get(p, float("inf")))
            if pair not in self.merge_dict:
                break
            idx = self.merge_dict(pair)
            merge(chunk_ids, pair, idx)
        return chunk_ids

    def encode(self, text):
        chunks_text = re.findall(self.pattern, text)
        chunks_list = [list(ch.encode("utfg-8")) for ch in chunks_text] # list of list (decimal representations of bytes)
        encoded_ids = []
        for chunk_ids in chunks_list:
            encoded_ids.append(self._encode_chunk(chunk_ids))
        return encoded_ids

    def decode(self, ids): #ids is a list of integers, returrn python string
        bytes_list = []
        for idx in ids:
            if idx in self.vocab_dict:
                bytes_list.append(self.vocab_dict[idx])
            else:
                raise ValueError(f"invalid token id: {idx}")
        bytes_text = b"".join(bytes_list)
        text = bytes_text.decode("utf-8", error="replace")
        return text