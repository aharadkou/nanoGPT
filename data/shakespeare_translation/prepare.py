import os
import numpy as np
import tiktoken
import random
import urllib

enc = tiktoken.get_encoding("gpt2")
space_encoded = enc.encode(" ")[0]
max_seq_len = 100

def tokenize(input_str, output_str):
   input_encoded = enc.encode(input_str)[:max_seq_len]
   output_encoded = enc.encode(output_str)[:max_seq_len]

   res = input_encoded + [enc.eot_token] + output_encoded

   seq_len = max_seq_len * 2 + 1
   spaces_len = 0 if len(res) >= seq_len else seq_len-len(res)

   return res + [space_encoded] * spaces_len, len(input_encoded)

def get_tokenized_result(input_data, output_data):
   input_str_len = []
   result = []
  
   for i in range(len(output_data)):
    tokenized, input_tokenized_len = tokenize(input_data[i], output_data[i])
    result += [tokenized]
    input_str_len += [input_tokenized_len]

   return result, input_str_len

def get_x_y(tokenized_array, input_str_len_arr):
   result = [[], []]

   for i in range(len(tokenized_array)):
    # x = tokenized_array[i][:-1]
    # y = tokenized_array[i][1:]

    # for i in range(input_str_len_arr[i] - 1):
    #    y[i] = -100

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

text_output, text_input = zip(*texts)

n_train = len(text_output)
train_size = 0.9
train_input_data = text_input[:int(n_train*train_size)]
val_input_data = text_input[int(n_train*train_size):]

train_output_data = text_output[:int(n_train*train_size)]
val_output_data = text_output[int(n_train*train_size):]


train_ids = np.array(get_x_y(*get_tokenized_result(train_input_data, train_output_data)))
val_ids = np.array(get_x_y(*get_tokenized_result(val_input_data, val_output_data)))


np.save(os.path.join(os.path.dirname(__file__), 'train'), train_ids)
np.save(os.path.join(os.path.dirname(__file__), 'val'), val_ids)
