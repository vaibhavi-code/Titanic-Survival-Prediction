import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression

import warnings

warnings.filterwarnings( 'ignore' )





# Importing and Reading the Data #


Training_Dataset = pd.read_csv( "C:/Users/Vaibhavi Nayak/Desktop/Project Dataset/train.csv" )

Testing_Dataset = pd.read_csv( "C:/Users/Vaibhavi Nayak/Desktop/Project Dataset/test.csv" )

Sample_Submission = pd.read_csv( "C:/Users/Vaibhavi Nayak/Desktop/Project Dataset/gender_submission.csv" )







# Data Pre-processing #


# 1. Handling the Missing Data in the Training Dataset and the Testing Dataset #


# Now , Training Dataset #

Training_Dataset[ 'Age' ].fillna( Training_Dataset[ 'Age' ].median() , inplace = True )


# We would calculate the Most Frequent Character in the Embarked Column / Mode #

# print( Training_Dataset[ 'Embarked' ].value_counts() )

# Output  : S : 644 , C : 168 , Q : 77

Training_Dataset[ 'Embarked' ].fillna( 'S' , inplace = True )



# Now , Testing Dataset #

Testing_Dataset[ 'Age' ].fillna( Testing_Dataset[ 'Age' ].median() , inplace = True )

Testing_Dataset[ 'Fare' ].fillna( Training_Dataset[ 'Fare' ].median() , inplace = True )





# Dropping Irrelevant Columns #


# Dropping Cabin , Ticket , Name , Sex_Male Columns from the Training Dataset #

Columns = [ 'Name' , 'Cabin' , 'Ticket' ]

Training_Dataset.drop( Columns , axis = 1 , inplace = True )

# print( Training_Dataset.columns )


# Dropping Cabin , Ticket , Name , Sex_Male Columns from the Testing Dataset #

Columns = [ 'Name' , 'Cabin' , 'Ticket' ]

Testing_Dataset.drop( Columns , axis = 1 , inplace = True )

# print( Testing_Dataset.columns )



 
# 2. Handling Categorical Data #


# Converting the Categorical Data into Numerical Data #


Training_Dataset = pd.get_dummies( Training_Dataset )

Testing_Dataset = pd.get_dummies( Testing_Dataset )


# print( Training_Dataset.columns ) 

# print( Testing_Dataset.columns ) 





# 3. Feature Selection #


Training_Dataset[ 'TravelAlone' ] = np.where( ( Training_Dataset[ 'SibSp' ] + Training_Dataset[ 'Parch' ] ) > 0 , 0 , 1 )

Testing_Dataset[ 'TravelAlone' ] = np.where( ( Testing_Dataset[ 'SibSp' ] + Testing_Dataset[ 'Parch' ] ) > 0 , 0 , 1 )

Columns = [ 'SibSp' , 'Parch' ]

Training_Dataset.drop( Columns , axis = 1 , inplace = True )

Testing_Dataset.drop( Columns , axis = 1 , inplace = True )


# print( Training_Dataset.columns ) 

# print( Testing_Dataset.columns ) 





# Deploying the Logistic Regression Model #


X_Training_Dataset = Training_Dataset.drop( 'Survived' , axis = 1 )

Y_Training_Dataset = Training_Dataset[ 'Survived' ]

X_Testing_Dataset = Testing_Dataset



Model = LogisticRegression()

Model.fit( X_Training_Dataset , Y_Training_Dataset )

Y_Testing_Dataset = Model.predict( X_Testing_Dataset )

Accuracy = round( Model.score( X_Training_Dataset , Y_Training_Dataset ) * 100 , 2 )

print( Accuracy )

print()





# Checking the accuracy of the Logistic Regression Model #

Training_Accuracy = pd.DataFrame( { 'Model': [ 'Logistic Regression' ] , 'Score': [ Accuracy ] } ) 

print( Training_Accuracy )






