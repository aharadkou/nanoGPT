import os
import numpy as np
import tiktoken
import random
import urllib

enc = tiktoken.get_encoding("gpt2")
space_encoded = enc.encode(" ")[0]
max_seq_len = 100

def tokenize(input_str, output_str):
   return tokenize_str(input_str) + tokenize_str(output_str)

def tokenize_str(str):
   max_len = max_seq_len - 1
   encoded = enc.encode(str)[:max_len]
   spaces_len = 0 if len(encoded) == max_len else max_len - len(encoded)

   return encoded + [enc.eot_token] + spaces_len * [space_encoded]

def get_tokenized_result(input_data, output_data):
   result = []
  
   for i in range(len(output_data)):
    tokenized = tokenize(input_data[i], output_data[i])
    result += [tokenized]

   return result

def get_x_y(tokenized_array):
   result = [[], []]

   for i in range(len(tokenized_array)):
    result[0].append(tokenized_array[i])
    result[1].append(tokenized_array[i])

   return result


def download_file_if_not_exist(file_name, url):
  file_path = os.path.join(os.path.dirname(__file__), file_name)

  if not os.path.exists(file_path):
      data_url = url
      with open(file_path, 'w', encoding="utf-8") as f:
          f.write(urllib.request.urlopen(data_url).read().decode("utf-8", "ignore"))

  with open(file_path, 'r', encoding="utf-8") as f:
    data = f.read()

  return data


input_file_name = 'input.txt'
output_file_name = 'output.txt'

input_data = download_file_if_not_exist(input_file_name, "https://raw.githubusercontent.com/emukans/shakespeare-texts/master/all_original.txt")
output_data = download_file_if_not_exist(output_file_name, "https://raw.githubusercontent.com/emukans/shakespeare-texts/master/all_modern.txt")

texts = list(zip(input_data.splitlines(), output_data.splitlines()))

random.shuffle(texts)

text_input, text_output = zip(*texts)

n_train = len(text_output)
train_size = 0.9
train_input_data = text_input[:int(n_train*train_size)]
val_input_data = text_input[int(n_train*train_size):]

train_output_data = text_output[:int(n_train*train_size)]
val_output_data = text_output[int(n_train*train_size):]


train_ids = np.array(get_x_y(get_tokenized_result(train_input_data, train_output_data)))
val_ids = np.array(get_x_y(get_tokenized_result(val_input_data, val_output_data)))


np.save(os.path.join(os.path.dirname(__file__), 'train'), train_ids)
np.save(os.path.join(os.path.dirname(__file__), 'val'), val_ids)
