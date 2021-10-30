import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("stopwords")

class TfIdfPreprocess:

    def filter(self, text):
        '''
        Text filter function. Removes mentions and other things from tweets
        Remove some extra stuff for BOW model
        Src: https://skimai.com/fine-tuning-bert-for-sentiment-analysis/
        '''
        lines = []

        for line in text.values:
            final_text = ''

            # TODO: change this to regular expressions as below
            for word in line.split():
                if word.startswith('@'):
                    continue
                elif word[-3:] in ['com', 'org']:
                    continue
                elif word.startswith('pic') or word.startswith('http') or word.startswith('www'):
                    continue
                else:
                    final_text += word + ' '
            
            line = line.lower()
            # Change 't to 'not'
            line = re.sub(r"\'t", " not", line)
            # Remove @name
            line = re.sub(r'(@.*?)[\s]', ' ', line)
            # Isolate and remove punctuations except '?'
            line = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', line)
            line = re.sub(r'[^\w\s\?]', ' ', line)
            # Remove some special characters
            line = re.sub(r'([\;\:\|•«\n])', ' ', line)
            # Remove stopwords except 'not' and 'can'
            line = " ".join([word for word in line.split()
                        if word not in stopwords.words('english')
                        or word in ['not', 'can']])
            # Remove trailing whitespace
            line = re.sub(r'\s+', ' ', line).strip()

            lines.append(final_text)

        return lines


    def feature_extraction(self, text):
        '''
        Feature extraction
        src: https://skimai.com/fine-tuning-bert-for-sentiment-analysis/
        '''
        tfidf = TfidfVectorizer(max_features=768)

        filtered = self.filter(text)

        x_tfidf = tfidf.fit_transform(filtered).toarray()
        
        assert(x_tfidf.shape[0] == len(filtered))
        
        return x_tfidf