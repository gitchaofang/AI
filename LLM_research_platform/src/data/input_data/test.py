path = "/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/src/data/input_data/data/novel.txt" 
with open(path, "r", encoding="utf-8") as f:
	text = f.read()
for word in text.split():
	print(f"{word}")
