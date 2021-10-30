import numpy as np
from transformers import AutoTokenizer, TFAutoModel


class BertPreprocess:

    def __init__(self, bert_model='bert-base-uncased', sentence_embedding_averaged=False):
        self.bert_model = bert_model
        self.sentence_embedding_averaged = sentence_embedding_averaged

    
    def filter(self, text):
        '''
        Text filter function. Removes mentions and other things from tweets
        TODO: we can finetune this function further
        '''
        lines = []

        for line in text.values:
            final_text = ''
            # line = str(line)
            for word in line.split():
                if word.startswith('@'):
                    continue
                elif word[-3:] in ['com', 'org']:
                    continue
                elif word.startswith('pic') or word.startswith('http') or word.startswith('www'):
                    continue
                else:
                    final_text += word + ' '
            lines.append(final_text)

        return lines


    def feature_extraction(self, text):
        '''
        Feature extraction
        '''
        batch_size = 100
        tokenizer = AutoTokenizer.from_pretrained(self.bert_model)
        bert = TFAutoModel.from_pretrained(self.bert_model)

        tokenizer.model_max_len = 512
        tokenizer.model_max_length = 512
     
        filtered = self.filter(text)

        X = []

        # Process the input in batches to speed up computation, but limit GPU memory usage
        for i in range(0, len(filtered), batch_size):
            end = i + batch_size if i + batch_size < len(filtered) else len(filtered)
            print(i, end)

            batch = filtered[i:end]

            # Call the BERT tokenizer. Use padding and truncation to enable batch processing. TODO: does this affect performance?
            encoded_x = tokenizer(batch, padding=True, truncation=True, return_tensors='tf') 
            
            # Process the data by using a pre-trained BERT model to extract usefull features
            output = bert(encoded_x)
            
            # Extract the last hidden state of the token `[CLS]` for classification task
            #TODO: this is from https://skimai.com/fine-tuning-bert-for-sentiment-analysis/. why?
            output2 = None
            
            if self.sentence_embedding_averaged:
                output2 = output[0][:, : , :].numpy()
                output2 = list(np.average(output2, 1)) #take average over all iinputs to get a document embedding 
            else:
                # Extract the last hidden state of the token `[CLS]` for classification task
                output2 = list(output[0][:, 0, :].numpy())
                
            X.extend(output2)
        
        assert(len(X) == len(filtered))
        
        return X
