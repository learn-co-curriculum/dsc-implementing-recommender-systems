
# Implementing Recommendation Engines with Surprise
## Introduction

This lesson will give you a brief introduction to implementing recommendation engines with a python library called surprise. In this lesson, you'll get a chance to try out multiple different types of collaborative filtering engines, ranging from both basic neighborhood-based methods to matrix factorization methods.

## Objectives
You will be able to:

* Compare and evaluate different recommendation algorithms
* Use surprise's built-in reader class to process data to work with recommender algorithms
* Cross-validate recommender algorithms
* Obtain predictions for a specific user at a specific point in time

In this lesson, we'll be working with a dataset built-in to surprise called the jester dataset. This dataset contains jokes rated by users on a scale of -10 to 10 based off a user's perceived humor level for a given joke. Although this is a fairly silly domain, you could understand how this might be important.


```python
from surprise import Dataset
from surprise.model_selection import train_test_split
```

First, you'll have to load the jokes dataset. You might have to download it first if prompted. Let's investigate the dataset after we load that. In this folder, you'll find the file for the text of jokes if you want to investigate what caliber of human you're dealing with here.


```python
jokes = Dataset.load_builtin(name='jester')
```


```python
type(jokes)
```




    surprise.dataset.DatasetAutoFolds




```python
# split into train and test set.
trainset, testset = train_test_split(jokes,test_size=0.2)
```

Notice how there is no X_train or y_train in our values here. Our only features here are the ratings of other users and items, so we need to keep everything together. What is happening in the train test split here is that surprise is randomly selecting certain $r_{ij}$ for users $u_{i}$ and items $i_{j} $ at the rate of 80% of the ratings in the train set and 20% in the test set. Let's investigate `trainset` and `testset` further.


```python
print('Type trainset :',type(trainset),'\n')
print('Type testset :',type(testset))
```

    Type trainset : <class 'surprise.trainset.Trainset'> 
    
    Type testset : <class 'list'>


Interestingly enough, the values here are different datatypes! The trainset is still a surprise specific datatype that is optimized for computational efficiency and the test set is a standard python list. You'll see why when we start making predictions. Let's take a look at how large our testset is as well as what's contained in an individual element. A sacrifice of surprise's implementation is that we lose a lot of the exploratory methods that are present with pandas.


```python
print(len(testset))
print(testset[0])
```

    352288
    ('11015', '147', 13.062)


## Memory-Based Methods (Neighborhood-Based)

To begin with, we can calculate the more simple neighborhood-based approaches. Some things to keep in mind are what type of similarities you should use. These can all have fairly substantial effects on the overall performance of the model. You'll notice that the API of surprise is very similar to sklearn when it comes to model training and testing.


```python
from surprise.prediction_algorithms import knns
from surprise.similarities import cosine, msd, pearson
from surprise import accuracy
```

One of our first decisions is item-item similarity versus user-user similarity. For the sake of computation time, it's best to calculate the similarity between whichever number is fewer, users or items. Let's see what the case is for our training set.


```python
print('Number of users: ',trainset.n_users,'\n')
print('Number of items: ',trainset.n_items,'\n')

```

    Number of users:  58763 
    
    Number of items:  140 
    


There are clearly way more users than items! We'll take that into account when inputting the specifications to our similarity metrics.


```python
sim_cos = {'name':'cosine','user_based':False}
```

Now it's time to train our model. Note that if you decide to train this model with user_based = True, it will take quite some time!


```python
basic = knns.KNNBasic(sim_options=sim_cos)
basic.fit(trainset)
```

    Computing the cosine similarity matrix...
    Done computing similarity matrix.





    <surprise.prediction_algorithms.knns.KNNBasic at 0x12ec3db70>



And now our model is fit! Let's take a look at the similarity metrics of each of the items to one another by using the `sim` attribute of our fitted model.


```python
basic.sim
```




    array([[1.        , 0.80218868, 0.81938356, ..., 0.82103422, 0.80772598,
            0.81253432],
           [0.80218868, 1.        , 0.82201347, ..., 0.78947946, 0.78136876,
            0.80547341],
           [0.81938356, 0.82201347, 1.        , ..., 0.85845477, 0.88341048,
            0.88940796],
           ...,
           [0.82103422, 0.78947946, 0.85845477, ..., 1.        , 0.91286961,
            0.87638776],
           [0.80772598, 0.78136876, 0.88341048, ..., 0.91286961, 1.        ,
            0.90604771],
           [0.81253432, 0.80547341, 0.88940796, ..., 0.87638776, 0.90604771,
            1.        ]])



Now it's time to test the model to determine how well our model performed


```python
predictions = basic.test(testset)
```


```python
print(accuracy.rmse(predictions))
```

    RMSE: 4.5057
    4.505684958221031


Not a particularly amazing model.... As you can see, the model had an RMSE of about 4.5, meaning that it was off by roughly 4 points for each guess it made for ratings. Not horrendous when you consider we're working on a range of 20 points, but let's see if we can improve it. To begin with, let's try with a different similarity metric and evaluate our RMSE.


```python
sim_pearson = {'name':'pearson','user_based':False}
basic_pearson = knns.KNNBasic(sim_options=sim_pearson)
basic_pearson.fit(trainset)
predictions = basic_pearson.test(testset)
print(accuracy.rmse(predictions))
```

    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    RMSE: 4.2799
    4.279931073999833


Here we are trying to minimize this objective function:

∑rui∈Rtrain(rui−(μ+bu+bi))2+λ(b2u+b2i).



```python
sim_pearson = {'name':'pearson','user_based':False}
knn_means = knns.KNNWithMeans(sim_options=sim_pearson)
knn_means.fit(trainset)
predictions = knn_means.test(testset)
print(accuracy.rmse(predictions))
```

    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    RMSE: 4.1360
    4.135960456406754


A little better... let's try one more neighborhood based method before moving into more model-based methods. Let's try the KNNBaseline method. This is a cool method because it adds a bias term that is calculated by way of minimizing for a cost function.


```python
sim_pearson = {'name':'pearson','user_based':False}
knn_baseline = knns.KNNBaseline(sim_options=sim_pearson)
knn_baseline.fit(trainset)
predictions = knn_baseline.test(testset)
print(accuracy.rmse(predictions))
```

    Estimating biases using als...
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    RMSE: 4.1313
    4.131267007009025


Even better! Now let's see if we can get some insight by applying some matrix factorization techniques!

## Model-Based Methods (Matrix Factorization)

It's worth pointing out that when SVD is calculated for recommendation systems, it is usually done with a modified version called "Funk's SVD" that only takes into account the rated values, ignoring whatever items have not been rated by users. The algorithm is named after Simon Funk, who was part of the team who placed 3rd in the Netflix challenge with this innovative way of performing matrix decomposition. Read more about Funk's SVD implementation at [his original blog post](https://sifter.org/~simon/journal/20061211.html). There is no simple way to include for this fact with scipy's implementation of svd, but luckily the surprise library has Funk's version of SVD implemented to make our lives easier!

It did perform better! As with other sklearn libraries, we can expedite the process of trying out different parameters by using an implementation of GridSearch. Let's make use of the Gridsearch here to account for some different configurations of parameters within the SVD pipeline. This might take some time! You'll notice that the n_jobs parameter set to -1, which ensures that all of the cores on my computer will be used to process fitting and evaluating all of these models. To help keep track of what is occurring here, take note of the different values. This code ended up taking over 16 minutes to complete even with parallelization in effect, so the optimal parameters are given to you. Use them to train a model and let's see how well it performs. If you want the full GridSearch experience, feel free to uncomment the code and give it a go!

The optimal parameters are :

```python
{'n_factors': 100, 'n_epochs': 10, 'lr_all': 0.005, 'reg_all': 0.4}

```


```python
from surprise.prediction_algorithms import SVD
from surprise.model_selection import GridSearchCV

# param_grid = {'n_factors':[20,100],'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
#               'reg_all': [0.4, 0.6]}
# gs_model = GridSearchCV(SVD,param_grid=param_grid,n_jobs = -1,joblib_verbose=5)
# gs_model.fit(jokes)
```

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  10 tasks      | elapsed:  1.3min
    [Parallel(n_jobs=-1)]: Done  64 tasks      | elapsed: 11.2min
    [Parallel(n_jobs=-1)]: Done  80 out of  80 | elapsed: 16.7min finished



```python
svd = SVD(n_factors=100,n_epochs=10,lr_all=0.005,reg_all=0.4)
svd.fit(trainset)
predictions = svd.test(testset)
print(accuracy.rmse(predictions))
```

    RMSE: 4.2945
    4.29447422093157


Interestingly, this model performed worse than the others! In general, the advantages of matrix factorization starts to show itself when the size of the dataset becomes massive. At that point, the storage challenges increase for the memory-based models, and there is enough data for latent factors to become extremely apparent.

## Making Predictions

Now that we've explored some models, let's use the most effective one to make some predictions on code. 

You might be wondering, "OK I'm making predictions about certain items rated by certain users, but how can I actually give a certain N recommendations to an individual user?" Although surprise is a great library, it does not have this recommendation functionality built into it, but in the next lab, you will get some experience not only fitting recommendation system models, but also programmatically retrieving recommended items for each user.

### Sources

Jester dataset originally obtained from:

[Eigentaste](http://www.ieor.berkeley.edu/~goldberg/pubs/eigentaste.pdf): A Constant Time Collaborative Filtering Algorithm. Ken Goldberg, Theresa Roeder, Dhruv Gupta, and Chris Perkins. Information Retrieval, 4(2), 133-151. July 2001.



## Summary

You now should have an understanding of the basics of how the library works. In the upcoming lab, you will be tasked with fitting models using surprise and then retrieving those predicted values in a meaningful way to give recommendations to people. Let's see how well it works in action.
