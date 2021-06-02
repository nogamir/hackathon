import re
import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize

text = 'My name is Matan 204 Hi %@'.lower()
text = re.sub('[^A-Za-z]', ' ', text)
text = word_tokenize(text)
print(text)
