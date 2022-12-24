import nltk

class BaseSubstitutor():
    def __init__(self):
        self.stopwords = nltk.corpus.stopwords.words("english")
    def get_subword(self, word, *args):
        return [], [], []
    def get_neighbor_list(self, word_list, *args):
        substitute_for_sentence = []
        probs_for_sentence= []
        orig_word_probs = []
        for x in word_list:
            sub_words_for_token, probs_for_token, orig_token_prob = self.get_subword(word_list, args)
            substitute_for_sentence.append(sub_words_for_token)
            probs_for_sentence.append(probs_for_token)
            orig_word_probs.append(orig_token_prob)
        return substitute_for_sentence, probs_for_sentence, orig_word_probs            

