class Tokenizer:
	def __init__(self,file_namd):
		path = "/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/src/data/input_data/data" + self.file_name
		with open(path, "r", encoding="utf-8") as f:
			text = f.read()
		words = set()
		for word in text.split():
			words.add(word)
		word_set = sorted(words)
		stoi = {
			i + 1: word 
			for i, word in enumerate(words)
		}
		itos = {
			word: i + 1
			for i, word in enumerate(words)
		}
		print(f"vocab_size is: {len(stoi)}")
		return {
			"stoi": self.stoi,
			"itos": self.itos
		}
		
