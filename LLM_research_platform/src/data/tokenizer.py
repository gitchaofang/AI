class Tokenizer:
	def __init__(self,file_name):
		path = "/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/src/data/input_data/data/" + file_name
		with open(path, "r", encoding="utf-8") as f:
			text = f.read()
		words = set()
		for word in text.split():
			words.add(word)
		word_set = sorted(words)
		self.stoi = {
			i + 1: word 
			for i, word in enumerate(words)
		}
		self.itos = {
			word: i + 1
			for i, word in enumerate(words)
		}
		print(f"vocab_size is: {len(self.stoi)}")
	def get():
		return {
			"stoi": self.stoi,
			"itos": self.itos
		}
		
