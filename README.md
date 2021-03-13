# Toy Sentiment Analysis

Questo codice ha solo valenza didattica e lo uso nelle mie conferenze divulgative sul NLP.

## BERT

BERT è acronimo di Bidirectional Encoder Representations from Transformers.

Di nuovo questo algoritmo è stato inventato nei laboratori di Google, nel 2018.

- Bidirezionale in quanto il contesto di una parola è preso sia dalle parole che la precedono che da quelle che la seguono;
- Encoder perché utilizza il tipico schema dei transformer, che infatti sono anche citati nell’acronimo.

Un sistema bidirezionale va a “mascherare” alcune delle parole nelle frasi del corpus per addestrarsi a predirle: per esempio nella frase

```
Arrivederci e grazie per tutto il pesce
```

potremmo mascherare tutto e pesce, ottenendo lo schema
```
Arrivederci e grazie per X il Y
```

Abbiamo così una frase cui associamo una label (X=tutto, Y=pesce) creando in questo modo un training record.

Naturalmente in questo modo possiamo creare moltissime combinazioni  e quindi un training set smisurato sul quale apprendere a predire le parole dato il contesto.

Un altro meccanismo di BERT consiste nel predire se due frasi sono collegate. Per esempio

```
Non temo l’intelligenza della macchina. Temo la stupidità dell’Uomo
```

sono due frasi chiaramente correlate, mentre

```
Non temo l’intelligenza della macchina. Che tempo fa?
```

non lo sono.

Per ottenere questo risultato BERT si esercita su enormi moli di frasi correlate e non.

Le caratteristiche di BERT (e degli algoritmi transformer in generale) richiedono che per addestrarli sui voluminosi training set sia richiesta una strumentazione particolare, cioè macchine potenti che consentano calcoli paralleli e siano in grado di svolgerli velocemente.

Per questo BERT separa nettamente la fase di training da quella di tuning che consiste nel trovare la configurazione giusta dell’algoritmo per uno specifico problema: per la prima ci si affida a un pre-training svolto da Google sui suoi server, la seconda ce la possiamo fare in casa.

## Una semplice (e imprecisa) sentiment analysis

Tentiamo un semplice esperimento di sentiment analysis, cioè capire se una frase sta dicendo qualcosa di positivo o di negativo.

Per farlo consideriamo un celebre dataset di recensioni di film prese dal sito Rotten Tomatoes. Si tratta di un database di 10605 giudizi, molto brevi (la maggior parte di una sola frase) su dei film, e un sentiment ranking per ognuno di essi, cioè un punteggio fra 0 e 1 tanto più alto quanto il giudizio è positivo e tanto più basso quanto è negativo

Alcune frasi del training set e relativi punteggi:

- “This is a movie where the most notable observation is how long you 've been sitting still” [0.27778 😠]
- “this one is a sweet and modest and ultimately winning story” [0.84722 😊]
- “and not worth” [0.43056 😐]

Per semplicità lavoreremo solo sulle singole frasi e utilizzeremo un modello BERT pre-addestrato, adattandolo alle esigenze del problema: ne scegliamo uno semplice, "BERT-tiny". Questo modello molto piccolo vuole in input 64  numeri e ne restituisce 128 (e internamente ha 4.369.152 parametri!).

La codifica del risultato del modello BERT per determinare il sentiment la faremo con una semplicissima rete neurale un layer di dropout e un layer denso da 128 neuroni di input per un totale di 16512 parametri interni

Provate a usare il programma e soprattutto a modificarlo!

Enjoy,
P
