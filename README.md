# Customer churn

## Задача оттока клиентов из банка
#### Репозиторий содержит материалы для решения задачи оттока. 

## Метериалы
* requirements.txt — список необходимых библиотек для решения задачи. 
Для установки всех библиотек выполните команду:

  `pip3 install -r requirements.txt`

* Churn_Modelling.csv — данные.
* 1-Exploratory_data_analysis.ipynb — основной анализ данных. 
* visualization.py — модуль для визуализации данных.
* doc_visualization.html — документация для visualization.py.

## Использование shell скриптов для других проектов 
Скрипты, которые есть в репозитории, можно использовать для генерации документаций .py файлов и для генерации requirements.txt для других проектов.

* get_py.sh — создает py файл на основе определенного jupyter notebook.
* get_req.sh — создает requirements.txt на основе всех файлов в проекте.
* get_doc.sh — создает документацию в формате html для конкретного файла. 

### ! Замечания
1. Если в проекте есть файлы формата .ipynb, нужно сначала создать на основе них .py файлы, а только после этого создавать requirements.txt.
2. Перед запуском скриптов нужно выполнить команды:      
  `pip3 install pipreqs`   
  `pip3 install pdoc3`
