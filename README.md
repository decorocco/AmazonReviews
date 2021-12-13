# Projeto Final: Megadados
### André Rocco, Beatriz Muniz, Marcelo C. Miguel

## Tarefa 1

- Quantos reviews existem?
      
      Existem 150.962.278 reviews nesse csv.

- Quantos clientes existem?
      
      Existem 33.497.620 customers diferentes

- Quantos produtos existem?
      
      Existem 21.390.118 produtos

- Quantos reviews existem para cada “star_rating” (de 1 a 5 estrelas)?
      
      +-----------+--------+
      |star_rating|   count|
      +-----------+--------+
      |          1|12099424|
      |          2| 7304329|
      |          3|12133772|
      |          4|26223155|
      |          5|93199322|
      +-----------+--------+


## Tarefa 2

### Para caracterizar bots no database executamos os seguintes passos em nosso Notebook:


- Cálculo da média de reviews por cliente

      Decidimos utilizar o dobro da média como referência para um numéro elevado de reviews por cliente

- Filtrar usuários por `Verified_Purchase == 'N'`

       Essa ferramenta (verified_purchase) foi criada pela amazon justamente para tentar filtrar bots de seus sistemas de reviews.
       Um bot dificilmente teria sua compra de um produto verificada pela amazon, já que são feitos por Web-Scraping e o agente responsável pelo bot não vincularia uma compra a cada conta, como não compra realmente os produtos.

- Identificar bots

      Se um  cliente tem mais do que o dobro da média de reviews e não possui compras verificadas, pode-se assumir que se trata de um bot.

- Numero de bots:

      Chegamos a um número de __ bots

## Tarefa 3

### Para criar o classificador executamos os seguintes passos em nosso Notebook:

- Foi feito um `na.drop` para limpar o dataset de dados vazios

      df_star_rating_clean = df_star_rating_clean.na.drop()

- Criação de uma coluna `type_review`, classificando os reviews como positivos, negativos ou neutros
  
      +-----------+--------+
      |type_review|   count|
      +-----------+--------+
      |   negativa|31533900|
      |     neutra|26220257|
      |   positiva|93187206|
      +-----------+--------+

- `regexTokenizer`

      regexTokenizer = RegexTokenizer(inputCol="review_body", outputCol="tokens", pattern="\\W+")
      stages += [regexTokenizer]

- `CountVectorize`

      cv = CountVectorizer(inputCol="tokens", outputCol="token_features", minDF=2.0)#, vocabSize=3, minDF=2.0
      stages += [cv]

- `Conversão de labels em valores numéricos`

      indexer = StringIndexer(inputCol="type_review", outputCol="label")
      stages += [indexer]

- `Vetorização de features`

      vecAssembler = VectorAssembler(inputCols=['token_features'], outputCol="features")
      stages += [vecAssembler]

- Fazendo o `fit do pipeline`

      pipeline = Pipeline(stages=stages)
      data = pipeline.fit(df_star_rating_clean).transform(df_star_rating_clean)

- `Divisão do database` em 70% treino e 30% teste

      train, test = data.randomSplit([0.7, 0.3], seed = 2018)

- Implementação do `Naive-Bayes`

      # Initialise the model
      nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
      # Fit the model
      model = nb.fit(train)
      # Make predictions on test data
      predictions = model.transform(test)
      predictions.select("label", "prediction", "probability").show()

- **Predição**

      +-----+----------+--------------------+
      |label|prediction|         probability|
      +-----+----------+--------------------+
      |  0.0|       0.0|[0.86108673188318...|
      |  1.0|       1.0|[2.35425484080207...|
      |  2.0|       0.0|[0.77663630503116...|
      |  0.0|       0.0|[0.83367010839255...|
      |  0.0|       0.0|[0.99996697214199...|
      |  0.0|       0.0|[0.98251929353495...|
      |  2.0|       0.0|[0.94946227287511...|
      |  0.0|       2.0|[1.85929937693236...|
      |  1.0|       1.0|[0.31916994842804...|
      |  0.0|       0.0|[0.96665233716123...|
      |  0.0|       0.0|[0.64850016169020...|
      |  0.0|       0.0|[0.98839926741271...|
      |  0.0|       0.0|[0.88775829951555...|
      |  0.0|       0.0|[0.82234737300667...|
      |  0.0|       0.0|[0.99370740918331...|
      |  2.0|       0.0|[0.54613827492242...|
      |  2.0|       2.0|[0.03987000712271...|
      |  0.0|       0.0|[0.99995148364510...|
      |  0.0|       0.0|[0.99698529397317...|
      |  0.0|       2.0|[0.15657256637537...|
      +-----+----------+--------------------+

- **Acurácia**

      A acurácia obtida foi de: