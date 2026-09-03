class Tokenizer:
	def __init__(self,file_name):
		#path = "/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/src/data/input_data/data/" + file_name # for local
		path = "/content/AI/LLM_research_platform/src/data/input_data/data/" + file_name # for codlab online # for codelab online
		with open(path, "r", encoding="utf-8") as f:
			text = f.read()
		chars = sorted(list(set(text)))
		print(f"chars: {chars} size is {len(chars)}")
		
		
		self.stoi = {
			char: i + 1 
			for i, char in enumerate(chars)
		}
		self.itos = {
			i + 1: char
			for i, char in enumerate(chars)
		}
	
	def get(self):
		return {
			"stoi": self.stoi,
			"itos": self.itos
		}
		
