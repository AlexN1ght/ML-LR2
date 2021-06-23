# Отчет по лабораторной работе 
## по курсу "Искусственый интеллект"

## Алгоритмы классификации


### Студент: 

Цапков А.М.

 
## Результат проверки

| Преподаватель         | Дата         |  Оценка       |
|-----------------------|--------------|---------------|
| Ахмед Самир Халид     |              |               |


## Тема работы

Реализовать 3 алгоритма классификации:
1) Логистическая Регрессия
2) SVM
3) Дерево решений

Провести анализ и вычислить метрики для своей реализации, а так же сравнить с ревлизацией в sklearn.

## Датасет
В качестве датаcета используется датасет с звездами и их типами из 1-й лабораторной работы. Он загружается из заранее подготовленных numpy файлов:
```
Star_train = np.load('Dataset/Star_train.npy')
Star_test = np.load('Dataset/Star_test.npy')
SLables_train = np.load('Dataset/SLables_train.npy')
SLables_test = np.load('Dataset/SLables_test.npy')
```


## Логистическая Регрессия

Для реализации данного алгоритма я создал класс с основными функциями fit, predict и score. Однако была проблема в том что изначально данный
алгоритм -- это алгоритм **бинарной** классификации, а мой датасет имеет 6 различных классов. Для решения этой проблемы я создал класс **MultiCalssificator**
который принимает в конструкторе бинарный классификатор и делает из него многоклассовый. Для этого он внутри себя делает несколько бинарных классификаторов 
сравнивая 1-й класс с остальными, а затем так же рекурсивно разделяет и остальные. Из-за такой реализации результативность классификатора зависит от 
порядка разбиения классоы, поэтому он задается в параметрах к методу fit, так же как и количество итераций.

### Теперь посмотрим на результаты:
#### Собственная реализация
```
classifier = MultiCalssificator(selfLogisticRegression)
classifier.fit(Star_train, SLables_train, [0, 1, 2, 3, 4, 5], 30000)
print(classifier.score(Star_test, SLables_test))
disp = plot_confusion_matrix(classifier, Star_test, SLables_test,
                                cmap=plt.cm.Blues,
                                normalize='true')
plt.show()
```
Вывод:
```
0.9625
```
![image](https://github.com/AlexN1ght/ML-LR2/blob/main/images/LogisticRegressionSelf.png)
#### SkLearn
```
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(max_iter=10000)
classifier.fit(Star_train, SLables_train)
print(classifier.score(Star_test, SLables_test))
disp = plot_confusion_matrix(classifier, Star_test, SLables_test,
                                cmap=plt.cm.Blues,
                                normalize='true')
plt.show()
```
Вывод:
```
0.8625
```
![image](https://github.com/AlexN1ght/ML-LR2/blob/main/images/LogisticRegressionSk.png)

## SVM
С этим алгоритмом была такая же проблема, что и с линейной регрессией, поэтому здесь мы тоже воспользуемся классом **MultiCalssificator**. Пораметр у 
функции fit-- количество эпох

### Результаты:
#### Собственная реализация
```
classifier = MultiCalssificator(selfSVM)
classifier.fit(Star_train, SLables_train, [0, 1, 2, 3, 4, 5], 2000)
print(classifier.score(Star_test, SLables_test))
disp = plot_confusion_matrix(classifier, Star_test, SLables_test,
                                cmap=plt.cm.Blues,
                                normalize='true')
plt.show()
```
Вывод:
```
1.0
```
![image](https://github.com/AlexN1ght/ML-LR2/blob/main/images/SVMself.png)
#### SkLearn
```
from sklearn import svm
classifier = svm.SVC()
classifier.fit(Star_train, SLables_train)
print(classifier.score(Star_test, SLables_test))
disp = plot_confusion_matrix(classifier, Star_test, SLables_test,
                                cmap=plt.cm.Blues,
                                normalize='true')
plt.show()
```
Вывод:
```
0.95
```
![image](https://github.com/AlexN1ght/ML-LR2/blob/main/images/SVMsk.png)

## Дерево решений
Данный алгоритм изначально может быть реализован многоклассово, поэтому нам не нужно прибегать к помощи отдельного класса. Параметры в конструкторе это 
максимальная глубина дерева и минимальное количество вхождений в 1 узле.
### Результаты:
#### Собственная реализация
```
classifier = selfDecisionTreeClassifier(10, 3)
classifier.fit(Star_train, SLables_train)
print(classifier.score(Star_test, SLables_test))
disp = plot_confusion_matrix(classifier, Star_test, SLables_test,
                                cmap=plt.cm.Blues,
                                normalize='true')
plt.show()
```
Вывод:
```
0.9875
```
![image](https://github.com/AlexN1ght/ML-LR2/blob/main/images/DecisionTreeClassifierSelf.png)
#### SkLearn
```
from sklearn import tree
classifier = tree.DecisionTreeClassifier()
classifier.fit(Star_train, SLables_train)
print(classifier.score(Star_test, SLables_test))
disp = plot_confusion_matrix(classifier, Star_test, SLables_test,
                                cmap=plt.cm.Blues,
                                normalize='true')
plt.show()
```
Вывод:
```
0.9875
```
![image](https://github.com/AlexN1ght/ML-LR2/blob/main/images/DecisionTreeClassifierSk.png)

## Выводы
Для логистической регрессии и SVM результаты моей реализации оказались выше. В SVM это связано с тем что я насильно делаю 2000 эпох обучения, а в sklern SVM 
можно специализировать только максимальное количество эпох. Однако результаты не очень сильно отличаются 1.0 и 0.95 соответственно. В Логистической же регрессии 
точность моей реализации сильно превышает точность sklearn 0.9625 против 0.8625. Это скорее всего связано с тем как мы реализуем многоклассовость. Я отделяю все 
классы по одному в определенном порядке так, чтобы каждый из них был линейно разделим с остальными. Многоклассовая классификация SVM в sklearn же проходит "под
капотом", однако мы можем четко заметить как данный класиификатор путает 0 и 1 классы. В случае с деревом решений результаты полностью идентичны 0.9875. Такой же
результат показывал алгоритм KNN в предыдущей лабораторной. По итогу лучшим классификатором оказалась моя реализация SVM с 2000 эпох, но при этом и самой долгой в 
обучении.
