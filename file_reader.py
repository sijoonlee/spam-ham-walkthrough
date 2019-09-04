import glob
import numpy as np
import random
import torch

class File_reader(object):
  def __init__(self):
    self.ham = []
    self.spam = []
    self.ham_paths = ["enron1/ham/*.txt", "enron2/ham/*.txt", "enron3/ham/*.txt", "enron4/ham/*.txt", "enron5/ham/*.txt", "enron6/ham/*.txt"]
    self.spam_paths = ["enron1/spam/*.txt", "enron2/spam/*.txt", "enron3/spam/*.txt", "enron4/spam/*.txt", "enron5/spam/*.txt", "enron6/spam/*.txt"]

  def read_file(self, path, minimum_word_count = 3, unnecessary =  ["-", ".", ",", "/", ":", "@"]):
    files  = glob.glob(path)
    content_list = []
    for file in files:
        with open(file, encoding="ISO-8859-1") as f:
            content = f.read()
            if len(content.split()) > minimum_word_count:
              content = content.lower()
              if len(unnecessary) is not 0:
                  content = ''.join([c for c in content if c not in unnecessary])
              content_list.append(content)
    return content_list

  def cut_before_combine(self, data, max = 5000):
    if max is not 0:
      if len(data) > max:
        random.shuffle(data)
        data = data[:max]
    return data

  def load_ham_and_spam(self, ham_paths = "default", spam_paths = "default", max = 5000): # 0 for no truncation

    if ham_paths == "default":
      ham_paths = self.ham_paths
    if spam_paths == "default":
      spam_paths = self.spam_paths

    self.ham = [ item for path in ham_paths for item in self.read_file(path) ]
    if max != 0:
      self.ham = self.cut_before_combine(self.ham, max)
    print("ham length ", len(self.ham))

    self.spam = [item for path in spam_paths for item in self.read_file(path) ]
    if max != 0:
      self.spam = self.cut_before_combine(self.spam, max)
    print("spam length ", len(self.spam))

    data = self.ham + self.spam

    ham_label = [0 for _ in range(len(self.ham))]
    spam_label = [1 for _ in range(len(self.spam))]

    label_tensor = torch.as_tensor(ham_label + spam_label, dtype = torch.int16)

    return data, label_tensor

  def print_sample(self, which ="both"): # ham, spam or both
    if which == "ham" or which == "both":
      idx = random.randint(0, len(self.ham))
      print("----------- ham sample -------------")
      print(self.ham[idx])
    if which == "spam" or which == "both":
      idx = random.randint(0, len(self.spam))
      print("----------- spam sample -------------")
      print(self.spam[idx])
