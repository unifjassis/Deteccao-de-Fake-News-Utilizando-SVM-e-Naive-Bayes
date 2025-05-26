# 📰 Detecção de Fake News com Naive Bayes e SVM

Este projeto tem como objetivo detectar **notícias falsas** utilizando **algoritmos de aprendizado de máquina supervisionado**, com foco nos modelos **Naive Bayes** e **Support Vector Machine (SVM)** aplicados a dados textuais.

## 📌 Tecnologias utilizadas

- Python 3
- Pandas e NumPy
- NLTK (pré-processamento)
- Scikit-learn (modelagem e avaliação)
- Matplotlib (visualização)

## 📁 Estrutura do Projeto

fake_news_classifier/  
├── data/ # Datasets originais  
│ └── ISOT/  
│ ├── True.csv  
│ └── Fake.csv  
├── src/ # Módulos principais  
│ ├── data_loader.py  
│ ├── preprocessor.py  
│ ├── vectorizer.py  
│ ├── trainer_nb.py  
│ ├── trainer_svm.py  
│ ├── evaluator.py  
│ └── comparator.py  
├── main.py # Pipeline principal  
└── requirements.txt # Dependências  


## 📊 Etapas do pipeline

1. Leitura e junção dos dados (fake e real)
2. Pré-processamento textual (limpeza, tokenização, stopwords)
3. Vetorização com TF-IDF
4. Treinamento com:
   - `Multinomial Naive Bayes`
   - `Linear SVM`
5. Avaliação com métricas: **acurácia**, **precisão**, **recall** e **f1-score**
6. Validação cruzada com 5-fold
7. Geração de gráfico comparativo

## 📚 Dataset utilizado  
ISOT Fake News Dataset - Kaggle


## 📈 Exemplo de saída  

Acurácia: 0.945
Precisão: 0.94
Recall:   0.95
F1-score: 0.945

## 🚀 Como executar

```bash
# Instale as dependências
pip install -r requirements.txt

# Execute o pipeline principal
python main.py
