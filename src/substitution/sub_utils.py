from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn

class BaseFilter():
    def filter_antonym(self, orig_word, word_list):
        return word_list

class MyLemmatizer():
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.pos_list = ['n','v','a','r','s']
    def lemmatize(self, word):
        lemma_list = [self.lemmatizer.lemmatize(word, x) for x in self.pos_list]
        filtered_lemma_list = [x for x in lemma_list if x != word]
        if word not in filtered_lemma_list:
            filtered_lemma_list.append(word)
        return filtered_lemma_list

class WordNetFilter(BaseFilter):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def filter_antonym(self, orig_word, word_list, word_pos):
        if word_pos == 'none':  ## skip the antonym filter process if POS is none
            return [i for i in range(len(word_list))]
        lemma_orig_word = self.lemmatizer.lemmatize(orig_word, pos = word_pos)
        lemma_word_list = [self.lemmatizer.lemmatize(x, pos = word_pos) for x in word_list]
        filtered_index_list = []
        
        assert len(lemma_word_list) == len(word_list)

        antonym_list = []
        for syn in wn.synsets(lemma_orig_word):
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    for antonym in lemma.antonyms():
                        antonym_list.append(antonym.name())
        
        for i in range(len(lemma_word_list)):
            if lemma_word_list[i] in antonym_list:
                continue
            filtered_index_list.append(i)
        return filtered_index_list



