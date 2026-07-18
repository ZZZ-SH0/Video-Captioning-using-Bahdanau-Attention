import config
import os
from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model
from tensorflow.keras.layers import AdditiveAttention
from tensorflow.keras.layers import Concatenate
import joblib


def inference_model():
    """Returns the model that will be used for inference"""
    with open(os.path.join(config.save_model_path, 'tokenizer' + str(config.num_decoder_tokens)), 'rb') as file:
        tokenizer = joblib.load(file)
    # loading encoder model. This remains the same
    inf_encoder_model = load_model(os.path.join(config.save_model_path, 'encoder_model.h5'))

    #updated architechture added attention, inference decoder model loading
    decoder_inputs = Input(shape=(None, config.num_decoder_tokens))

    decoder_state_input_h = Input(shape=(config.latent_dim,))
    decoder_state_input_c = Input(shape=(config.latent_dim,))

    encoder_outputs_input = Input(shape=(config.time_steps_encoder,config.latent_dim))

    decoder_states_inputs = [
        decoder_state_input_h,
        decoder_state_input_c
    ]

    decoder_lstm = LSTM(
        config.latent_dim,
        return_sequences=True,
        return_state=True
    )

    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs,initial_state=decoder_states_inputs)

    attention_layer = AdditiveAttention()

    context_vectors = attention_layer([decoder_outputs, encoder_outputs_input])

    decoder_combined = Concatenate(axis=-1)([decoder_outputs, context_vectors])

    decoder_dense = Dense(
        config.num_decoder_tokens,
        activation='softmax'
    )

    decoder_outputs = decoder_dense(
        decoder_combined
    )

    inf_decoder_model = Model([decoder_inputs,encoder_outputs_input,decoder_state_input_h,decoder_state_input_c],[decoder_outputs,state_h,state_c])
    
    #Load model
    inf_decoder_model.load_weights(os.path.join(config.save_model_path,'decoder_model.weights.h5'))
    return tokenizer, inf_encoder_model, inf_decoder_model


