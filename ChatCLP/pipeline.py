import torch
import transformers
model = transformers.GPT2LMHeadModel.from_pretrained('gpt2')

def generate_response(input_str):
  tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
  input_ids = torch.tensor(tokenizer.encode(input_str)).unsqueeze(0)  # Encode the input string as a tensor
  attention_mask = torch.ones_like(input_ids)  # Create an attention mask
  pad_token_id = tokenizer.pad_token_id  # Get the pad token id
  output = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=pad_token_id, max_length=1024, top_p=0.9, top_k=0)  # Generate a response
  return tokenizer.decode(output[0], skip_special_tokens=True)  # Decode the response and remove special tokens

while True:
  input_str = input("You: ")
  if input_str == 'exit':
    break
  response = generate_response(input_str)
  print("AI:", response)
