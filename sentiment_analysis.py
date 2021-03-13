import bert
import numpy as np
import os

print("Scansione del training set...")

BERT_PATH = "bert_layer"
VOCAB_TXT = os.path.join(BERT_PATH, "vocab.txt")

DATASET_DIR = "stanfordSentimentTreebank"
SENTENCES = os.path.join(DATASET_DIR, "datasetSentences.txt")
SCORES = os.path.join(DATASET_DIR, "sentiment_labels.txt")
SPLITTING = os.path.join(DATASET_DIR, "datasetSplit.txt")
DICTIONARY = os.path.join(DATASET_DIR, "dictionary.txt")

MAX_LENGTH = 64    # Length of word vectors which the model accepts as input

tokenizer = bert.bert_tokenization.FullTokenizer(VOCAB_TXT, do_lower_case = True)

training_set, training_scores = [], []
testing_set, testing_scores = [], []

print("Definisce il training e il testing set...")

def read_file(filename):
    with open(filename) as f:
        f.readline()            # skips the heading line
        return f.readlines()

sentences = read_file(SENTENCES)
scores = read_file(SCORES)
splitting = read_file(SPLITTING)
dictionary = read_file(DICTIONARY)

# let scores[i] = score being i an int index and score a float score.
scores = {int(s[:s.index("|")]): float(s[s.index("|")+1:]) for s in scores}

# let splitting[i] = int denoting the kind of dataset (1=training, 2=testing).
splitting = {int(s[:s.index(",")]): int(s[s.index(",")+1:]) for s in splitting}

# let dictionary[s] = phrase index of the corresponding string
dictionary = {s[:s.index("|")]: int(s[s.index("|")+1:]) for s in dictionary}

print("Prepara i dati nel formato richiesto da BERT...")
# Now looks for each sentence inside the dictionary, retrieves the index and
# looks for the index in the scores, creating a list of sentences and scores
for s in sentences:
    i = int(s[:s.index("\t")])       # sentence index, to be matched in splitting
    s = s[s.index("\t") + 1:][:-1]   # extract the sentence (strip the ending "\n")
    if s not in dictionary:
        continue
    ph_i = dictionary[s]             # associated phrase index
    # Now tokenizes the sentence and put it into the BERT format
    s = tokenizer.tokenize(s)
    if len(s) > MAX_LENGTH - 2:
        s = s[:MAX_LENGTH - 2]
    s = tokenizer.convert_tokens_to_ids(["[CLS]"] + s + ["[SEP]"])
    if len(s) < MAX_LENGTH:
        s += [0] * (MAX_LENGTH - len(s))
    # Decides in which dataset to store the data
    if splitting[i] == 1:
        training_set.append(s)
        training_scores.append(scores[ph_i])
    else:
        testing_set.append(s)
        testing_scores.append(scores[ph_i])

training_set, training_scores = np.array(training_set), np.array(training_scores)
testing_set, testing_scores = np.array(testing_set), np.array(testing_scores)

print("Carica il modello pre-allenato da Google")

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling1D, Input, Lambda
from tensorflow.keras.optimizers import Adam

# Loads the bert pre-trained layer to plug into our network
bert_params = bert.params_from_pretrained_ckpt(BERT_PATH)
bert_layer = bert.BertModelLayer.from_params(bert_params, name = "BERT")
bert_layer.apply_adapter_freeze()

print("Definisce la rete neurale transformer e ne stampa la struttura...")

# We arrange our layers by composing them as functions, with the input layer as inmost one
input_layer = Input(shape=(MAX_LENGTH,), dtype = 'int32', name = 'input_ids')
output_layer = bert_layer(input_layer)

output_layer = GlobalAveragePooling1D()(output_layer)
output_layer = Dense(128, activation = "relu")(output_layer)
output_layer = Dropout(0.1)(output_layer)
output_layer = Dense(1, activation = "relu")(output_layer)

neural_network = Model(inputs = input_layer, outputs = output_layer)
neural_network.build(input_shape = (None, MAX_LENGTH))
neural_network.compile(loss = "mse", optimizer = Adam(learning_rate = 3e-5))

neural_network.summary()

try:
    print("Se ho salvato in precedenza la rete la carico da file...")
    neural_network.load_weights("pesi_rete")
except:
    print("No! La alleno e poi la salvo...")
    neural_network.fit(
        training_set,
        training_scores,
        batch_size= 128,
        shuffle = True,
        epochs = 4,
        validation_data = (testing_set, testing_scores),
        verbose = 1
    )
    neural_network.save_weights("pesi_rete")

print("Inserire (in inglese) delle frasi con giudizi su film:")
while True:
    s = input(">")
    if s == "":
        break
    t = tokenizer.tokenize(s)
    if len(t) > MAX_LENGTH - 2:
        t = t[:MAX_LENGTH - 2]
    t = tokenizer.convert_tokens_to_ids(["[CLS]"] + t + ["[SEP]"])
    if len(t) < MAX_LENGTH:
        t += [0] * (MAX_LENGTH - len(t))
    p = neural_network.predict(np.array([t]))[0][0]
    print("Sentiment:", p)

##print("Ora propongo alcune frasi e le ordino dalla peggiore alla migliore")
##
##some_sentences = [
##    "The film is not bad but the actors should take acting lessons",
##    "Another chiefwork by a master of western movies",
##    "This film is just disappointing: do not waste time on it",
##    "Well directed but poorly acted",
##    "The movie is well directed and greatly acted",
##    "A honest zombie movie with actually no new ideas",
##]
##for s in some_sentences:
##    print("\t", s)
##ranked = []   # pairs (rank, sentence)
##for s in some_sentences:
##    t = tokenizer.tokenize(s)
##    if len(t) > MAX_LENGTH - 2:
##        t = t[:MAX_LENGTH - 2]
##    t = tokenizer.convert_tokens_to_ids(["[CLS]"] + t + ["[SEP]"])
##    if len(t) < MAX_LENGTH:
##        t += [0] * (MAX_LENGTH - len(t))
##    p = neural_network.predict(np.array([t]))[0][0]
##    ranked.append((p, s))
##print("Ordinate dalla più negativa alla più positiva")
##for r in sorted(ranked):
##    print("\t", r[1])
