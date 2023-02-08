# basic libs
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# cleaning data
import re
import os
# import nltk
# nltk.download("stopwords")
# nltk.download('punkt')

# save vocabulary in files
import pickle

# pad sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model,load_model

path=os.path.join("F:\Data Science\Projects\Machine_translation_ENG_FR\model","Model_english_to_frensh.h5")
model=load_model(path)

word_2_idx_target=pickle.load(open("F:\Data Science\Projects\Machine_translation_ENG_FR/vocabulary/word_2_idx_target.txt","rb"))
word_2_idx_input=pickle.load(open("F:\Data Science\Projects\Machine_translation_ENG_FR/vocabulary/word_2_idx_input.txt","rb"))

num_encoder_tokens=13902
max_encoder_sequence_len=47
num_decoder_tokens=26560
max_encoder_sequence_len=57

def make_references():
  encoder_input=Input(shape=(None,),name="encoder_input_layer")
  encoder_embedding=model.get_layer("encoder_embedding_layer")(encoder_input)
  encoder_lstm=model.get_layer("encoder_lstm_1_layer")(encoder_embedding)
  encoder_lstm2=model.get_layer("encoder_lstm_2_layer")(encoder_lstm)
  _,state_h,state_c=encoder_lstm2
  encoder_states=[state_h,state_c]
  encoder_reference_model=Model(encoder_input,encoder_states)


  decoder_state_h=Input(shape=(256,))
  decoder_state_c=Input(shape=(256,))
  decoder_input_states=[decoder_state_h,decoder_state_c]

  decoder_input=Input(shape=(None,),name="decoder_input_layer")
  decoder_embedding=model.get_layer("decoder_embedding_layer")(decoder_input)
  decoder_lstm=model.get_layer("decoder_lstm_layer")

  decoder_outputs,state_h,state_c=decoder_lstm(decoder_embedding,initial_state=encoder_states)
  decoder_dense=model.get_layer("deocer_final_layer")
  
  decoder_outputs,state_h,state_c=decoder_lstm(decoder_embedding,initial_state=decoder_input_states)

  decoder_state=[state_h,state_c]
  decoder_outputs=decoder_dense(decoder_outputs)
  decoder_reference_model=Model([decoder_input]+decoder_input_states,[decoder_outputs]+decoder_state)

  return encoder_reference_model,decoder_reference_model

# clean english column
def clean_english(text):
  text=text.lower() # lower case

  # remove any characters not a-z and ?!,'
  text=re.sub(u"[^a-z!?',]"," ",text)

  # word tokenization
  text=text.split(" ")

  # join text
  text=" ".join([i.strip() for i in text])
  return text
def prepare_text(text):
  text=clean_english(text)

  res=[word_2_idx_input[i] for i in text.split(" ")]
  pad=pad_sequences([res],maxlen=5,padding="post")
  return pad
# prepare_text("How are you")

def translate_eng_fr(text):
  enc_model,dec_model=make_references()
  
  try:
    states_value=enc_model(prepare_text(text))
    empty_target_seq=np.zeros((1,1))
    empty_target_seq[0,0]=word_2_idx_target["start"]

    stop_condition=False
    decoded_translaition=""

    while not stop_condition:
        dec_output,h,c=dec_model.predict([empty_target_seq]+states_value)
        sampled_word_index=np.argmax(dec_output[0,-1,:])
        sampled_word=None

        for word,index in word_2_idx_target.items():
            if sampled_word_index == index:
                decoded_translaition+=' {}'.format(word)
                sampled_word=word

            if sampled_word == "end" or len(decoded_translaition.split(" ")) >= 12:
                stop_condition=True

        empty_target_seq=np.zeros((1,1))
        empty_target_seq[0,0]=sampled_word_index
        states_value=[h,c]
    decoded_translaition=" ".join(decoded_translaition.split()[:-1])
    return decoded_translaition
  except:
      return "not recognized"