from pathlib import Path
import yaml
import regex as re

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

FILE_DIR = Path("/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/src/data/input_data/data/token_training/")
#FILE_DIR = Path("/content/AI/LLM_research_platform/src/data/input_data/data/token_training/")


# for tokenization:
# 0: for padding
# 1: for EOS
# tokenization starts from 2
PAD_ID = 0
EOS_ID = 1 
TOKEN_OFFSET = 2
class SimpleTokenizer:
	def __init__(self, file_dir = FILE_DIR, pattern = None):
		self.file_dir = Path(file_dir)
		# pattern
		self.pattern = GPT2_SPLIT_PATTERN if pattern is None else pattern
		self.compiled_pattern = re.compile(self.pattern)
		self.stoi = {}
		self.itos = {}
		
	def train(self):
		chars = set()
		for file_path in self.file_dir.glob("*.txt"):
			print(f"training on {file_path}")
			with open(file_path, "r", encoding = "utf-8") as f:
				text = f.read()
				chars.update(text)

		chars = sorted(list(chars))
		self.stoi = {
				char: i + TOKEN_OFFSET 
				for i, char in enumerate(chars)
			}
		self.itos = {
				i + TOKEN_OFFSET: char
				for i, char in enumerate(chars)
			}
	def _encode_chunk(self, chunk_text):
		ids = [self.stoi[char] for char in chunk_text]
		return ids
	
	def encode(self, text):
		chunks_text = self.compiled_pattern.findall(text)
		encoded_ids = []

		for chunk_text in chunks_text:
			encoded_ids.append(self._encode_chunk(chunk_text))

		# deal with EOS
		if encoded_ids:
			encoded_ids[-1].append(EOS_ID)
		else:
			encoded_ids.append([EOS_ID])
		return encoded_ids
	
	def decode(self, ids):
		return "".join(self.itos[id] for id in ids)		
	
	def get_vocab_size(self):
		return len(self.stoi)