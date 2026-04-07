[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/rERvMAgO)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=23461131&assignment_repo_type=AssignmentRepo)
# Projeto de PI - Parte 2 - Análise dos Efeitos

A parte 1 deste projeto foi a captura de fotos e implementação de diferentes efeitos para editá-las. A partir de todas as fotos e efeitos que a turma implementou, geramos um dataset aplicando cada efeito em todas as fotos. 

Quando recebemos um novo dataset para uma tarefa, muitas vezes o nosso primeiro impulso é implementar o modelo que realiza esta tarefa. Porém, um passo anterior e essencial na área de aprendizado de máquina é explorar os dados, entender o que eles são, como se comportam, quais são suas características. Esta etapa não apenas nos ajuda a entender qual será a melhor técnica a aplicar, como também a encontrar problemas nestes dados e a tratá-los de forma ética: será que estão completos? será que existe alguma falha nas anotações? será que existem vieses nestes dados, existe alguma tendência neles, no que representam ou em como foram anotados?

Embora seja um dataset simplificado, esta tarefa irá simular este processo exploratório.


## Tarefas

1. Baixe o dataset utilizando o link disponibilizado no Google Classroom
   - Coloque o dataset na pasta `photos`, ele tem um `.gitignore` para evitar enviar as fotos para o GitHub.
2. Implemente um código para ler o dataset (dica: utilize [Dataset e Dataloader](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html))
3. Treine um ou mais algoritmos supervisionados e não supervisionados utilizando o dataset
   - O objetivo dos algoritmos supervisionados deve ser classificar as fotos por tipo de efeito. Não exatamente por esta tarefa ser de interesse, mas porque o comportamento do modelo nela pode trazer informações interessantes sobre os dados.
   - O objetivo dos algoritmos não supervisionados será encontrar padrões nos dados.
   - Você pode utilizar técnicas como modelos lineares, CNN, KNN, PCA, entre outras.
   - Não é obrigatório, mas pode ser útil utilizar técnicas de augmentação (rotacionar as fotos, transladar as fotos, recortar trechos das fotos, entre outros)
4. Realize análises qualitativas e quantitativas sobre os modelos obtidos. A ideia é que você consiga descrever qual o comportamento do modelo e quais as propriedades do dataset causam este comportamento. Explore o dataset e modelos. Algumas perguntas que podem ajudar nessa análise:
   - Os modelos performam bem ou mal? Por quê?
   - Existem grupos de fotos que ele classifica de forma errada (supervisionado), ou que agrupa juntos (não supervisionado)? Por que isto acontece?
   - O que esses comportamentos nos dizem sobre os dados?
   - Além dos modelos, você pode realizar outras análises sobre os dados para auxiliar a analisá-los


## Formato da dataset

O dataset é organizado em pastas. O nome de cada pasta corresponde a uma label (efeito), e o nome de um arquivo corresponde a uma foto. O mesmo nome em pastas diferentes corresponde a mesma foto original.

## Entrega

A atividade deve ser realizada no notebook `Atividade.ipynb`, realizando o commit ao terminar a atividade. Certifique-se de enviar o notebook com as saídas de todas as células.
