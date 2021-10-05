############################################################################################################################################################################################

# Importing the Libraries #


import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression


############################################################################################################################################################################################


# Ignoring the Warnings #


import warnings

warnings.filterwarnings( 'ignore' )


############################################################################################################################################################################################


# Importing the Data #


Dataset = pd.read_csv( "C:/Users/Vaibhavi Nayak/Desktop/Project/2. Titanic Survival Prediction/Dataset.csv" )


############################################################################################################################################################################################


# Cleaning the Data #


# 1. Handling the Missing Data in the Dataset #


Dataset[ 'Age' ].fillna( Dataset[ 'Age' ].median() , inplace = True )

Dataset[ 'Fare' ].fillna( Dataset[ 'Fare' ].median() , inplace = True )

# print( Dataset[ 'Embarked' ].value_counts() )

# Output  : S : 644 , C : 168 , Q : 77

Dataset[ 'Embarked' ].fillna( 'S' , inplace = True )





# 2. Dropping Irrelevant Columns #


# Dropping Cabin , Ticket , Name Columns from the Dataset #

Columns = [ 'Name' , 'Cabin' , 'Ticket' , 'Embarked' ]

Dataset.drop( Columns , axis = 1 , inplace = True )





# 3. Handling Categorical Data #


# Converting the Categorical Data into Numerical Data #


Dataset = pd.get_dummies( Dataset )

Columns = [ 'Sex_female' ]

Dataset.drop( Columns , axis = 1 , inplace = True )


############################################################################################################################################################################################


# Feature Selection #


Dataset[ 'TravelAlone' ] = np.where( ( Dataset[ 'SibSp' ] + Dataset[ 'Parch' ] ) > 0 , 0 , 1 )

Columns = [ 'SibSp' , 'Parch' ]

Dataset.drop( Columns , axis = 1 , inplace = True )


############################################################################################################################################################################################


# Training the Model and Testing the Model #


y = Dataset[ 'Survived' ]

Columns = [ 'Survived' ]

x = Dataset.drop( Columns , axis = 1 )


# Splitting the Dataset #


from sklearn.model_selection import train_test_split

X_Training_Dataset , X_Testing_Dataset , Y_Training_Dataset , Y_Testing_Dataset = train_test_split( x , y , test_size = 0.2 , random_state = 0 )


Model = LogisticRegression()

Model.fit( X_Training_Dataset , Y_Training_Dataset ) # Training the Model #

Prediction = Model.predict( X_Testing_Dataset ) # Testing the Model #


############################################################################################################################################################################################


# Accuracy of the Model #


from sklearn.metrics import accuracy_score

Accuracy = accuracy_score( Y_Testing_Dataset , Prediction )

print( Accuracy )


############################################################################################################################################################################################

