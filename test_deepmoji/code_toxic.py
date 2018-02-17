# -*- coding: utf-8 -*-
import numpy as np

import nltk
from nltk.corpus import reuters # to load data set
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer


import keras
from keras.models import Model, Sequential
from keras.layers.merge import concatenate
from keras.layers import Input, Bidirectional, Embedding, Dense, Dropout, SpatialDropout1D, LSTM, Activation
from keras.regularizers import L1L2
from keras.models import load_model

## Building of the vocabulary
import json
from deepmoji.attlayer import AttentionWeightedAverage
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.model_def import deepmoji_feature_encoding
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
i
mport string
print('Library imported')



def load_vocab_prepare_data(sentences,vocab_path = VOCAB_PATH,maxlen =100):
	'''
		Va charger le vocabulaire de Deepmoji et preparer l'input du modèle à partir de l'embedding des textes
		INPUT : 
			vocab_path : Où trouver le vocabulaire, a priori on peut mettre ce que l'on veut mais on préfère garder le vocab ini
			maxlen : taille de l'espace de sortie pour chaque mots
		OUT: 
			renvoie le data set prêt à entrainer le modèle (stemmer + enlevement des stop words)
	'''
	stop_words = set(stopwords.words('english'))
	stmr = PorterStemmer()
	sentence_util = []
	for sentence in sentences:
		processed_sentence = []
		for word in sentence:
			if word not in stop_words and word not in string.punctuation:
				processed_sentence.append(stmr.stem(word))
		sentence_util.append(''.join(processed_sentence))


	print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
	with open(VOCAB_PATH, 'r') as f:
		vocabulary = json.load(f)
	st = SentenceTokenizer(vocabulary, maxlen)
	tokenized, _, _ = st.tokenize_sentences(sentence_util)
	return(tokenized)

def pre_processing(sentence):
    good_sentence = []
    stop_words = set(stopwords.words('english'))
    stmr = PorterStemmer()
    for word in sentence:
         if word not in stop_words and word not in string.punctuation:
            good_sentence.append(stmr.stem(word))
    good_sentence = ' '.join(good_sentence)
    return(good_sentence)

#nb_token = tokenized.max()+1

def test_architecture(nb_classes, nb_tokens, maxlen, embed_l2=1E-6, return_attention=False):
	"""
		Renvoie la structure du modèle

	# Arguments:
		nb_classes: Niombe de classe dans le Data set, a priori 90
		nb_tokens: taille vocabulary 
		maxlen: taille max d'un mot

		embed_l2: L2 regularization for the embedding layerl.

	# Returns:
		Model with the given parameters.
	"""
	# define embedding layer that turns word tokens into vectors
	# an activation function is used to bound the values of the embedding
	print('Beggining build model')
	model_input = Input(shape=(maxlen,), dtype='int32', name= 'Input First')
	print('Embedding reg')
	embed_reg = L1L2(l2=embed_l2) if embed_l2 != 0 else None
	print('Beggining embedding layer')
	embed = Embedding(input_dim=nb_tokens,
					  output_dim=256,
					  mask_zero=True,
					  input_length=maxlen,
					  embeddings_regularizer=embed_reg,
					  name='embedding')
	print('embeb finished')
	x = embed(model_input)
	print('x finished')
	x = Activation('tanh')(x)
	print('Finish introduction')

	# entire embedding channels are dropped out instead of the
	# normal Keras embedding dropout, which drops all channels for entire words
	# many of the datasets contain so few words that losing one or more words can alter the emotions completely

	# skip-connection from embedding to output eases gradient-flow and allows access to lower-level features
	# ordering of the way the merge is done is important for consistency with the pretrained model
	print('LSTM building')
	lstm_output = Bidirectional(LSTM(512, return_sequences=True), name="bi_lstm_0")(x)

	x = concatenate([lstm_output, x])

    # if return_attention is True in AttentionWeightedAverage, an additional tensor
    # representing the weight at each timestep is returned
	x = AttentionWeightedAverage(name='attlayer', return_attention=return_attention)(x)
	print('LSTMs ready')

	# if return_attention is True in AttentionWeightedAverage, an additional tensor
	# representing the weight at each timestep is returned
	weights = None
	#x = AttentionWeightedAverage(name='attlayer', return_attention=return_attention)(x)
	#print('Attention layer ready')

	outputs = [x]

	return Model(inputs=[model_input], outputs=outputs, name="riminder")



def predict_cat(train_set):
	to_predict = []
	for sentence in train_set:
		sentence_processed = pre_processing(sentence)
		to_predict.append(sentence_processed)

	to_predict, _, _ = st.tokenize_sentences(to_predict)
	prediction = model.predict(to_predict)
	return(prediction)

## training
def encode_categories(docs):
	'''
		Encode les cotégories en un hot-vector à partir de train_docs ou test_docs
		Input:
			docs : liste des documents, train ou test
		Output : 
			les catégories encodées
	'''
	categories = [reuters.categories(document) for document in docs]

	mlb = MultiLabelBinarizer()
	encoded_cat =  mlb.fit_transform(categories)
	return(encoded_cat)

def compile_fit(model,data,target, save = True):
	'''
		Réalise la compilation du modèle et réalise le fit du modele sur les données -> data
		Input : 
			model : model à compiler et entrainer
			data :  array tokenize par la function load_vocab_prepare_data
			target : array avec les catégories pre processed par la fonction encode_categories
		Output:
			model compilé et entrainé
	'''
	model.compile(optimizer = "Adam",loss = "categorical_crossentropy")
	model.fit(tokenized,encoded_cat,batch_size = 100,epochs = 10)
	if save:
		model.save('model_riminder_last')
	return(model)

def load_trained_model():
	'''
		Charge un modèle déjà entrainé
		Output : 
			model chargé
	'''
	model = load_model('model_riminder_good.h5',custom_objects={'AttentionWeightedAverage': AttentionWeightedAverage})
	return(model)



def compute_cosine(code_doc_1,code_doc_2):
	'''
		Calcul la similarité entre deux documents à partir de l'angle entre deux documents
		Input:
			code_doc_1 : code du document, qui correspond à son code dans la liste documents
			code_doc_2 : document 2
		Output:
			Le cosinus entre les deux docs
	'''
	text_1,_,_ = st.tokenize_sentences([pre_processing(list(reuters.words(code_doc_1)))])
	text_2,_,_ = st.tokenize_sentences([pre_processing(list(reuters.words(code_doc_2)))])
	pred_1 = model.predict(text_1)
	pred_2 = model.predict(text_2)
	return(cosine(pred_1,pred_2),text_1)
