# from uu import encode
# from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
# import json
# dataset = Dataset_Loader(dName="test", dDescription="testing", data_type="classification")
# dataset.load()

# data = dataset.data



from transformers import BertTokenizer

# Load BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example text
text = "bromwell high cartoon comedy ran time programs school life teachers 35 years teaching profession lead believe bromwell highs satire much closer reality teachers scramble survive financially insightful students see right pathetic teachers pomp pettiness whole situation remind schools knew students saw episode student repeatedly tried burn school immediately recalled high classic line inspector im sack one teachers student welcome bromwell high expect many adults age think bromwell high far fetched pity isnt"

# Tokenize with BERT tokenizer
bert_inputs = bert_tokenizer(text)

# print(bert_inputs)

# for i in range(len(bert_inputs['input_ids'])):
#     print(f"Token {i}: {bert_tokenizer.convert_ids_to_tokens(bert_inputs['input_ids'][i])}")


# print("Token IDs:", bert_inputs['input_ids'])

# attention_mask = bert_inputs['attention_mask']
# print("Attention Mask:", attention_mask)

# token_type_ids = bert_inputs['token_type_ids']
# print("Token Type IDs:", token_type_ids)

# # Print the tokens themselves to understand the splits
# tokens = bert_tokenizer.convert_ids_to_tokens(bert_inputs['input_ids'][0])
# print("Tokens:", tokens)