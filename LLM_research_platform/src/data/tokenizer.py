class Tokenizer:
	def __init__(self,file_name):
		#path = "/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/src/data/input_data/data/" + file_name # for local
		path = "/content/AI/LLM_research_platform/src/data/input_data/data/" + file_name # for codlab online # for codelab online
		with open(path, "r", encoding="utf-8") as f:
			text = f.read()
		words = set()
		for word in text.split():
			words.add(word)
		word_set = sorted(words)
		self.stoi = {
			word: i + 1 
			for i, word in enumerate(words)
		}
		self.itos = {
			i + 1: word
			for i, word in enumerate(words)
		}
	
	def get(self):
		return {
			"stoi": self.stoi,
			"itos": self.itos
		}
		
