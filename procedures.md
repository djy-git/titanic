# 1. Dataset
[Titanic](https://www.kaggle.com/c/titanic/data)

## 1.1. Features
| Variable	| Definition |	Key |
| :---: | :---: | :---: |
| survival	|Survival	|0 = No, 1 = Yes|
|pclass	|Ticket class|	1 = 1st, 2 = 2nd, 3 = 3rd|
|sex	|Sex	| |
|Age	|Age| in years|
|sibsp|	# of siblings / spouses aboard the Titanic|	|
|parch|	# of parents / children aboard the Titanic	||
|ticket|	Ticket number	||
|fare	|Passenger fare	||
|cabin|	Cabin number	||
|embarked|	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton||

## 1.2. Exploration
### 1.2.1. Data
![](images/2.jpg)

### 1.2.2. Nan values

```py
> train_data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB

> test_data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 418 entries, 0 to 417
Data columns (total 11 columns):
PassengerId    418 non-null int64
Pclass         418 non-null int64
Name           418 non-null object
Sex            418 non-null object
Age            332 non-null float64
SibSp          418 non-null int64
Parch          418 non-null int64
Ticket         418 non-null object
Fare           417 non-null float64
Cabin          91 non-null object
Embarked       418 non-null object
dtypes: float64(2), int64(4), object(5)
memory usage: 36.0+ KB
```

- Imputing 필요 variables
`Age` (20%):  <br>
`Cabin` (92%): column 제거 <br>
`Embarked` (2개): row 제거 <br>
`Fare` (1개): row 제거

### 1.2.3. Numerical variables
```py
> train_data.hist(bins=50, figsize=(20, 15));
```
![](images/1.png)
- `Fare`, `Parch`, `SibSp`: log scale로 변경
- `PassengerID`는 imputing 시에만 사용하고 예측할 땐 제거
- `Pclass`: dummy variables로 변경
