#*****************************************************************************
#
#   Prevendo a Inadimplência de Clientes com Machine Learning
#
#   Fonte: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
#
#   Autor: Matheus Lanna
#
#*****************************************************************************

# Instalando os pacotes que serão utilizados no projeto (Necessário apenas uma vez)
install.packages("Amelia")
install.packages("caret")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("reshape")
install.packages("randomForest")
install.packages("e1071")
install.packages("rstudioapi")
install.packages("DMwR")

# Carregando os pacotes instalados acima
library("Amelia")
library("caret")
library("ggplot2")
library("dplyr")
library("reshape")
library("randomForest")
library("e1071")
library("rstudioapi")
library("DMwR")

# Determinando a pasta de trabalho como a pasta onde se encontra o script
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

# Carregando o dataset a ser utilizado
# Fonte: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
dados_clientes <- read.csv("dados/dataset.csv")

# verificação básica de dados
View(dados_clientes)
dim(dados_clientes)
str(dados_clientes)
summary(dados_clientes)


### Análise Exploratória, limpeza e transformação dos dados a serem usados ###

# Removendo a primeira coluna ID, por ser desnecessária (RStudio já indexa o dataset)
dados_clientes$ID <- NULL

# Renomeando as colunas de classe para facilitar identificação
colnames(dados_clientes)[2] <- "Genero"
colnames(dados_clientes)[3] <- "Escolaridade"
colnames(dados_clientes)[4] <- "Estado_Civil"
colnames(dados_clientes)[5] <- "Idade"
colnames(dados_clientes)[24] <- "Inadimplente"

# Identificação de valores ausentes e remoção do dataset
sapply(dados_clientes, function(x) sum(is.na(x)))                # Soma de valores ausentes por coluna
missmap(dados_clientes, main = "Valores ausentes identificados") # Mapa de  valores ausentes
dados_clientes <- na.omit(dados_clientes)                        # Remoção de valores ausentes

# Formatação da coluna Gênero
summary(dados_clientes$Genero) 
dados_clientes$Genero <- cut(dados_clientes$Genero,              # Transforma o gênero "1" em masculino
                             c(0,1,2),                           # e o gênero "2" em feminino, 
                             labels = c("Masculino",             # conforme documentação do dataset.
                                        "Feminino"))
summary(dados_clientes$Genero) 

# Formatação da coluna Escolaridade
summary(dados_clientes$Escolaridade) 
dados_clientes$Escolaridade <- cut(dados_clientes$Escolaridade,  # "1" - pós-graduado, "2" - graduado
                             c(0,1,2,3,4),                       # "3" - ensino médio, "4" - outro
                             labels = c("Pos Graduado",          # conforme documentação do dataset.
                                        "Graduado",
                                        "Ensino Medio",
                                        "Outro"))
summary(dados_clientes$Escolaridade) 

# Formatação da coluna Estado Civil
summary(dados_clientes$Estado_Civil) 
dados_clientes$Estado_Civil <- cut(dados_clientes$Estado_Civil,  # Transforma "1" em casado
                             c(-1,0,1,2,3),                      #  "2" em solteiro, "3" em outro
                             labels = c("Desconhecido",                # conforme documentação do dataset.
                                        "Casado",
                                        "Solteiro",
                                        "Outro"))
summary(dados_clientes$Estado_Civil) 

# Convertendo a variável Idade para o tipo fator com criação de faixa etária
summary(dados_clientes$Idade) 
hist(dados_clientes$Idade)
dados_clientes$Idade <- cut(dados_clientes$Idade, 
                            c(0,30,50,100), 
                            labels = c("Jovem", 
                                       "Adulto", 
                                       "Idoso"))
summary(dados_clientes$Idade)

# Convertendo a variável que indica pagamentos para o tipo fator
dados_clientes$PAY_0 <- as.factor(dados_clientes$PAY_0)
dados_clientes$PAY_2 <- as.factor(dados_clientes$PAY_2)
dados_clientes$PAY_3 <- as.factor(dados_clientes$PAY_3)
dados_clientes$PAY_4 <- as.factor(dados_clientes$PAY_4)
dados_clientes$PAY_5 <- as.factor(dados_clientes$PAY_5)
dados_clientes$PAY_6 <- as.factor(dados_clientes$PAY_6)

# Verificação do dataset após transformações feitas
str(dados_clientes) 
sapply(dados_clientes, function(x) sum(is.na(x)))
missmap(dados_clientes, main = "Valores Faltantes Observados")
dados_clientes <- na.omit(dados_clientes)                      # Exclusão dos valores não identificados
missmap(dados_clientes, main = "Valores Faltantes Observados")
dim(dados_clientes)
View

# Alterando a variável dependente (Inadimplente) para o tipo fator
dados_clientes$Inadimplente <- as.factor(dados_clientes$Inadimplente)
table(dados_clientes$Inadimplente)                              # Total de inadimplentes vs não-inadimplentes
prop.table(table(dados_clientes$Inadimplente))                  # Porcentagem de inadimplentes vs não-inadimplentes
qplot(Inadimplente, data = dados_clientes, geom = "bar") +      # Distribuição  de inadimplentes vs não-inadimplentes
  theme(axis.text.x = element_text(angle = 90, hjust = 1))    

# Gerando números aleatórios para criar uma distribuição entre dados de treino e teste
set.seed(12345)

# Amostragem estratificada, separando entre treino e teste
# Seleciona as linhas de acordo com a variável Inadimplente como strata
indice <- createDataPartition(dados_clientes$Inadimplente, p = 0.75, list = FALSE)
dim(indice)

# Definição dos dados de treinamento como subconjunto do conjunto de dados original
# com números de indice de linha (conforme identificado acima) e todas as colunas
dados_treino <- dados_clientes[indice,]
dim(dados_treino)
table(dados_treino$Inadimplente)
prop.table(table(dados_treino$Inadimplente))                  # Porcentagem entre classes
dim(dados_treino)                          

# Comparação entre as porcentagens das classes de treinamento e dados originais
compara_dados <- cbind(prop.table(table(dados_treino$Inadimplente)), 
                       prop.table(table(dados_clientes$Inadimplente)))
colnames(compara_dados) <- c("Treinamento", "Original")
compara_dados

# Melt Data - Converte colunas em linhas para melhor visualização dos dados
melt_compara_dados <- melt(compara_dados)
melt_compara_dados

# Plot para ver a distribuição do treinamento vs original
ggplot(melt_compara_dados, aes(x = X1, y = value)) + 
  geom_bar( aes(fill = X2), stat = "identity", position = "dodge") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Carregando os dados de teste (usando os dados que não estão no treino)
dados_teste <- dados_clientes[-indice,]
dim(dados_teste)
dim(dados_treino)



##### Modelo de Machine Learning ######



# Construção do primeiro modelo
View(dados_treino)
modelo_v1 <- randomForest(Inadimplente ~ ., data = dados_treino)
modelo_v1

# Avaliação do modelo
plot(modelo_v1)

# Previsões com dados de teste
previsoes_v1 <- predict(modelo_v1, dados_teste)

# Confusion Matrix
cm_v1 <- caret::confusionMatrix(previsoes_v1, dados_teste$Inadimplente, positive = "1")
cm_v1

# Calculando Precision, Recall e F1-Score, métricas de avaliação do modelo preditivo
y <- dados_teste$Inadimplente
y_pred_v1 <- previsoes_v1

precision <- posPredValue(y_pred_v1, y)
precision

recall <- sensitivity(y_pred_v1, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

# Balanceamento de classe
table(dados_treino$Inadimplente)
prop.table(table(dados_treino$Inadimplente))
set.seed(9560)
dados_treino_bal <- SMOTE(Inadimplente ~ ., data  = dados_treino)                         
table(dados_treino_bal$Inadimplente)
prop.table(table(dados_treino_bal$Inadimplente))

# Construção a segunda versão do modelo
modelo_v2 <- randomForest(Inadimplente ~ ., data = dados_treino_bal)
modelo_v2

# Avaliando o modelo
plot(modelo_v2)

# Previsões com dados de teste
previsoes_v2 <- predict(modelo_v2, dados_teste)

# Confusion Matrix
cm_v2 <- caret::confusionMatrix(previsoes_v2, dados_teste$Inadimplente, positive = "1")
cm_v2

# Calculando Precision, Recall e F1-Score, métricas de avaliação do modelo preditivo
y <- dados_teste$Inadimplente
y_pred_v2 <- previsoes_v2

precision <- posPredValue(y_pred_v2, y)
precision

recall <- sensitivity(y_pred_v2, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

# Determinando a importância das variáveis preditoras para as previsões
View(dados_treino_bal)
varImpPlot(modelo_v2)

# Obtendo as variáveis mais importantes
imp_var <- importance(modelo_v2)
varImportance <- data.frame(Variables = row.names(imp_var), 
                            Importance = round(imp_var[ ,'MeanDecreaseGini'],2))

# Criando o rank de variáveis baseado na importância
rankImportance <- varImportance %>% 
  mutate(Rank = paste0('#', dense_rank(desc(Importance))))

# Usando ggplot2 para visualizar a importância relativa das variáveis
ggplot(rankImportance, 
       aes(x = reorder(Variables, Importance), 
           y = Importance, 
           fill = Importance)) + 
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank), 
            hjust = 0, 
            vjust = 0.55, 
            size = 4, 
            colour = 'yellow') +
  labs(x = 'Variables') +
  coord_flip() 

# Construindo a terceira versão do modelo apenas com as variáveis mais importantes
colnames(dados_treino_bal)
modelo_v3 <- randomForest(Inadimplente ~ PAY_0 + PAY_2 + PAY_3 + PAY_AMT1 + PAY_AMT2 + PAY_5 + BILL_AMT1, 
                          data = dados_treino_bal)
modelo_v3

# Avaliando o modelo
plot(modelo_v3)

# Previsões com dados de teste
previsoes_v3 <- predict(modelo_v3, dados_teste)

# Confusion Matrix
cm_v3 <- caret::confusionMatrix(previsoes_v3, dados_teste$Inadimplente, positive = "1")
cm_v3

# Calculando Precision, Recall e F1-Score, métricas de avaliação do modelo preditivo
y <- dados_teste$Inadimplente
y_pred_v3 <- previsoes_v3

precision <- posPredValue(y_pred_v3, y)
precision

recall <- sensitivity(y_pred_v3, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

# Salvando o modelo em disco
saveRDS(modelo_v3, file = "modelo/modelo_v3.rds")

# Carregando o modelo
modelo_final <- readRDS("modelo/modelo_v3.rds")

# Previsões com novos dados de 3 clientes

# Dados dos clientes
PAY_0 <- c(0, 0, 0) 
PAY_2 <- c(0, 0, 0) 
PAY_3 <- c(1, 0, 0) 
PAY_AMT1 <- c(1100, 1000, 1200) 
PAY_AMT2 <- c(1500, 1300, 1150) 
PAY_5 <- c(0, 0, 0) 
BILL_AMT1 <- c(350, 420, 280) 

# Concatena em um dataframe
novos_clientes <- data.frame(PAY_0, PAY_2, PAY_3, PAY_AMT1, PAY_AMT2, PAY_5, BILL_AMT1)
View(novos_clientes)

# Checando os tipos de dados
str(novos_clientes)

# Convertendo os tipos de dados
novos_clientes$PAY_0 <- factor(novos_clientes$PAY_0, levels = levels(dados_treino_bal$PAY_0))
novos_clientes$PAY_2 <- factor(novos_clientes$PAY_2, levels = levels(dados_treino_bal$PAY_2))
novos_clientes$PAY_3 <- factor(novos_clientes$PAY_3, levels = levels(dados_treino_bal$PAY_3))
novos_clientes$PAY_5 <- factor(novos_clientes$PAY_5, levels = levels(dados_treino_bal$PAY_5))
str(novos_clientes)

# Previsões
previsoes_novos_clientes <- predict(modelo_final, novos_clientes)
View(previsoes_novos_clientes)