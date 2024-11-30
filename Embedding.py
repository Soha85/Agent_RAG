import re
import torch
import spacy
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
import numpy
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
nlp = spacy.load("en_core_web_sm")

class Embedding:
    corpus_content = ""
    corpus_chunks = []

    def simple_normalize(self,text):
        text = re.sub(r'\W', ' ', text.lower())
        text = text.replace('\\n', '')
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        return text

    def get_synonyms(self,word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return synonyms

    def expand_with_synonyms(self,words):
        expanded_words = words.copy()
        for word in words:
            expanded_words.extend(self.get_synonyms(word))
        return expanded_words

    def spacy_preprocess(self,text):
        text = text.replace('\\n', '')
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        doc = nlp(text.lower())
        lemmatized_words = []
        real_words = []
        for token in doc:
            if  token.is_punct or token.is_sent_end:
                continue
            lemmatized_words.append(token.lemma_)
            real_words.append(token.text)
        return ' '.join(lemmatized_words), ' '.join(real_words)

    def get_bert_embeddings(self,text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()

    def chunk_text(self,text, chunk_size, overlap):
        words = text.split(' ')
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
        return chunks

    def prepare_data(self,chunk_size,overlap):
        for context in self.corpus_content:
            # Combine question and context (as one block of text)
            # Split the document into chunks
            chunks = self.chunk_text(context,chunk_size,overlap)
            self.corpus_chunks.extend(chunks)