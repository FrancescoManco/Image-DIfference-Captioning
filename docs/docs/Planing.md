# Progetto di Image Difference Captioning

## Obiettivo del Progetto
L'obiettivo è sviluppare un sistema di **Image Difference Captioning** in grado di prendere in ingresso una coppia di immagini simili e generare una descrizione delle differenze tra le due.

---

## Planning del Progetto

### Fase 1: Preparazione del Dataset

####  Generazione delle Caption di Base
- **Obiettivo**: Creare una didascalia descrittiva di ogni immagine nel dataset.
- **Strumenti**: Utilizza un modello di *Image Captioning* pre-addestrato (ad es., `BLIP`, `Show and Tell`, o modelli simili).
- **Output**: Ottieni un dataset strutturato con le coppie di immagini e le loro descrizioni individuali:
    ```plaintext
    (Immagine1, Caption1), (Immagine2, Caption2)
    ```

####  Generazione delle Caption delle Differenze
- **Obiettivo**: Creare una didascalia che descriva solo le differenze tra le immagini di ciascuna coppia.
- **Strumenti**: Usa un *Large Language Model* (ad es., LLaMA 3.2) per confrontare `Caption1` e `Caption2` ed estrapolare una differenza testuale.
- **Output**: Ottieni un dataset strutturato con triplette `(Immagine1, Immagine2, CaptionDifferenza)`:
    ```plaintext
    (Immagine1, Immagine2, CaptionDifferenza)
    ```

---

## Struttura del Dataset Finale

- `(Immagine1, Immagine2, CaptionDifferenza)`: Ogni tripla rappresenta una coppia di immagini con la relativa descrizione delle differenze.
    ```plaintext
    (Immagine1, Immagine2, CaptionDifferenza)
    ```
--- 
### Fase 2: Sviluppo del Modello

#### 2.1 Costruzione dell'Encoder con Rete Siamese
- **Obiettivo**: Implementare un encoder siamese che prenda in input una coppia di immagini e produca degli embedding distinti ma confrontabili.
- **Architettura**: Utilizza un backbone pre-addestrato (come ResNet o EfficientNet) per estrarre le caratteristiche. Assicurasi che entrambe le reti condividano i pesi.
- **Passaggio di Output**:
    - Sperimentare con due metodi di combinazione degli embedding:
        - **Differenza**: Calcola la differenza tra gli embedding delle due immagini.
        - **Concatenazione**: Concatenazione diretta degli embedding.

#### 2.2 Implementazione del Decoder
- **Opzione 1**: *Decoder LSTM/GRU con Attention*:
  - Implementare un meccanismo di attention per concentrarsi sulle aree differenziate.
- **Opzione 2**: *Decoder Transformer*:
  - Utilizzare una struttura Transformer per la generazione di testo.
- **Obiettivo Finale del Decoder**: Generare sequenzialmente la didascalia della differenza.

---

### Fase 3: Preparazione e Addestramento del Modello

#### 3.1 Definizione della Funzione di Perdita
- Utilizzare una funzione di perdita *Cross-Entropy* per valutare la correttezza delle didascalie generate.
- **Regolarizzazione con Similarità Coseno**: Aggiungi una penalità basata sulla similarità coseno per penalizzare output troppo simili tra di loro.

#### 3.2 Impostazione dell’Ambiente di Addestramento
- **Dataloader**: Costruire un dataloader per fornire le triple `(Immagine1, Immagine2, CaptionDifferenza)`.
- **Ottimizzatore e Scheduler**: Usare Adam o AdamW come ottimizzatore e imposta un learning rate scheduler.
- **Validazione**: Preparare un set di validazione per monitorare le performance e prevenire overfitting.

#### 3.3 Addestramento
- Avviare l’addestramento per iterazioni multiple, monitorando la convergenza della perdita e le metriche di valutazione.

---

### Fase 4: Valutazione e Fine-tuning del Modello

#### 4.1 Metriche di Valutazione
- **BLEU, ROUGE, METEOR**: Utilizzare queste metriche per valutare la qualità delle didascalie generate.
- **Valutazione Manuale**: Eseguire una valutazione qualitativa su un campione di output per garantire che il modello catturi le differenze desiderate.



---

### Fase 5: Integrazione e Applicazione Finale

#### 5.1 Deploy del Modello
- Integrare il modello addestrato in un’applicazione o API che consenta di inserire coppie di immagini e ricevere una descrizione delle differenze.


---



