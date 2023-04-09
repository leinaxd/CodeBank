# Types of models:

## Types of input/outputs models
    - Tensor
        -Ex: Classes, Vocabulary, Numeric Variables, parameters
    - Sequence
        -Ex: Frames, Symbols, Text
    - mapping
        -Ex: functions, probability distribution, regression, Continuous variables
    - other structure
        -Ex: Graphs, objects, instructions

## Encoder:
    Input: Sequence
    Output: tensor (vector)
    Ex: NER, Sentiment analysis, language models
## Decoder:
    Input: tensor (vecor)
    Output: Sequence
    Ex: Question answering, Translator
## Feedfoward:
    Input: Tensor
    Output: Tensor
    Ex: Functions, Classifiers
## Generative model:
    Input: Sequence
    Output: Sequence
    Ex: Markov, RNN, Kalman
## Regression:
    Input:  tensor
    Output: mapping (function)
    Ex: pricing prediction (regression), 
## Tasks:
    -Anomaly detection


## A estudiar:
    DotCSV: https://www.youtube.com/watch?v=B7CgWjf6HnE
    -Modelos generativos:
        Ej.
            -StyleGAN / stylegan para video
        -Vector Quantization: Formato Autoencoder. (comprimir + modificar para generar algo diferente)
            -obtiene un codebook de features
            -Con el codebook se puede introducir un metodo autoregresivo (Transformer) y completar la imagen (CLIP)
        -Modelos difusos:
            -
    -Modelo predictivo:
        -Tesla. 
        -Es mas robusto a equivocarse, pero tampoco aprende de la experiencia