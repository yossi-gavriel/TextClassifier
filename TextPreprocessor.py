import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import re
import string
import nltk


class TextPreprocessor:
    def __init__(self, lowercase=True, remove_punctuation=True, remove_numbers=True, remove_stopwords=True, stem_words=False, lemmatize_words=False):
        """
        Constructor for the TextPreprocessor class.

        Args:
        - lowercase: bool, whether to convert text to lowercase (default=True)
        - remove_punctuation: bool, whether to remove punctuation from text (default=True)
        - remove_numbers: bool, whether to remove numbers from text (default=True)
        - remove_stopwords: bool, whether to remove stopwords from text (default=True)
        - stem_words: bool, whether to stem words in text (default=False)
        - lemmatize_words: bool, whether to lemmatize words in text (default=False)
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.stem_words = stem_words
        self.lemmatize_words = lemmatize_words

        if remove_stopwords or stem_words or lemmatize_words:
            nltk.download('stopwords')
            nltk.download('wordnet')
            from nltk.corpus import stopwords
            from nltk.stem import PorterStemmer, WordNetLemmatizer
            self.stopwords = set(stopwords.words('english'))
            self.stemmer = PorterStemmer() if stem_words else None
            self.lemmatizer = WordNetLemmatizer() if lemmatize_words else None

        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = None

    def preprocess(self, text_data):
        """
        Preprocesses a list of text data.

        Args:
        - text_data: list, the list of text data to preprocess

        Returns:
        - preprocessed_data: list, the preprocessed text data
        """
        preprocessed_data = []
        for text in text_data:
            # Convert to lowercase
            if self.lowercase:
                text = text.lower()

            # Remove punctuation
            if self.remove_punctuation:
                text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

            # Remove numbers
            if self.remove_numbers:
                text = re.sub('\d+', '', text)

            # Remove stopwords
            if self.remove_stopwords:
                text = ' '.join([word for word in text.split() if word not in self.stopwords])

            # Stem words
            if self.stem_words:
                text = ' '.join([self.stemmer.stem(word) for word in text.split()])

            # Lemmatize words
            if self.lemmatize_words:
                text = ' '.join([self.lemmatizer.lemmatize(word) for word in text.split()])

            preprocessed_data.append(text)

        if self.vocab is None:
            self.vocab = build_vocab_from_iterator(map(self.tokenizer, preprocessed_data))

        preprocessed_data = [self.vocab(tokenized_text) for tokenized_text in map(self.tokenizer, preprocessed_data)]

        return preprocessed_data