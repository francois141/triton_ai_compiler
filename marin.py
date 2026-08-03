from transformers import AutoModelForCausalLM, AutoTokenizer
marin = AutoModelForCausalLM.from_pretrained("marin-community/marin-8b-base")
tokenizer = AutoTokenizer.from_pretrained("marin-community/marin-8b-base")
message = ["The Marin wind is"]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
response = marin.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
