import numpy as np
import pandas as pd
from data_handling import *
from collections import defaultdict
import sys
np.random.seed(1)

class gNaiveBayes():
    '''
    This is the general class for handling the naive bayes classification.
    Assumption:
    1. data is give in form of dataframe with last column as labels.
    2. additionaly the name of the attributes and the class labels
    '''
    ###############################################################
    ####################### CLASS ATTRIBUTES ######################
    #Data related variables
    train_df=None       #the train-dataframe with last columns as labels
    valid_df=None       #the valid-dataframe with last columns as labels
    #Attributes and Class related variables
    attr_vals=None      #dict of all attributes and their val possible
    class_vals=None     #list of all possible world in Hogwartz
    #Learned Attributes from training set
    world_prob=None     #dictionary containing prob of each world
    world_f_prob=None   #the cond. prob of feature in the world


    ###############################################################
    ####################### MEMBER FUNCTION #######################
    #Intializer function
    def __init__(self,class_vals,attr_vals,train_df,valid_df):
        '''
        Initialization function for the classifier.
        '''
        #Setting up the variables for the classifier
        self.class_vals=class_vals
        self.attr_vals=attr_vals
        self.train_df=train_df
        self.valid_df=valid_df

    #Function to calculate the world probability
    def get_worlds_probability(self):
        '''
        This function will calculate the probability of the world
        ie. different class labels which will be used to weigh the
        probabiliy of the point after we get its prob in that world.
        '''
        #Getting the total number of documents
        train_df=self.train_df
        total_examples = train_df.shape[0]

        #Initializing the world prob
        print("\n\nCalculating the probability of different class")
        world_prob={}
        #Iterating over the class values
        for cval in class_vals:
            #Counting the number of example of this class
            cval_count = train_df[train_df["labels"]==cval].shape[0]
            #Calculating the probability of the world
            world_prob[cval]=cval_count/total_examples
            print("prob:{0:.6f}\tcount:{1:.6f}\tworld's cval:{2}".format(
                                            world_prob[cval],
                                            cval_count,
                                            cval))
        #Assigning the world prob to class
        self.world_prob=world_prob

    #Function to calcualte the probability of each attribute in
    def get_attr_prob_in_world(self):
        '''
        This function will calculate the probability distribution of
        each attributes in each of the world
        '''
        #Initializing the features in each world's probability
        world_f_prob=defaultdict(lambda:defaultdict(defaultdict))
        train_df=self.train_df

        print("\n\nLearning the attribute prob distrbution in each world")
        #Iterating over each world
        for world,wframe in train_df.groupby("labels"):
            #Calculating the count of this world in dataset
            world_count=wframe.shape[0]
            #Iterating over all the attributes in this world
            for aval in self.attr_vals.keys():
                #Getting the number of attributes in this aval
                num_avals=len(list(self.attr_vals.keys()))
                #Now finding the probability of each of the attributes
                for fval in attr_vals[aval]:
                    fframe=wframe[wframe[aval]==fval]
                    #Getting the count of this particular val of attr
                    feature_count=fframe.shape[0]
                    #Now saving the probability of this feature
                    prob=(feature_count+1)/(world_count+num_avals)
                    world_f_prob[world][aval][fval]=prob
                    print("World:{}\tAttr:{}\tFeature:{}\tprob:{}"\
                                                .format(world,
                                                        aval,
                                                        fval,
                                                        prob))
                print("\n")
            print("\n")
        #Assigning the world prob to class
        self.world_f_prob=world_f_prob

    #Function to evaluate the accuracy on a particular dataset
    def evaluate_dataset(self,data_df):
        '''
        This function will run over all the points present in the
        dataset make prediction using the trained probaiblites of
        each attributes in every world.
        '''
        #Initializing the counters for confusion matrix (default 0)
        confusion_mat=defaultdict(lambda:defaultdict(int))

        print("Starting the prediction for given dataset.Hold Tight!")
        #Iterating over all the examples
        for idx in data_df.index.tolist():
            #Retreiving the points from the dataframe
            point=data_df.loc[idx]
            label=point["labels"]

            #Now getting the prediction label from our trained model
            # print("idx:\n",idx,point)
            pred_world,prob=self.make_prediction_point(point)
            # print("Doc:{0} \twith prob:{1:.6f} \tevaluated:{2}\tactually:{3}".\
            #                         format(idx,prob,pred_world,label))
            #Now adding the entry to the confusion matrix
            confusion_mat[label][pred_world]+=1

        #Printing the confusion matrix
        self._print_evaluation_metrics(confusion_mat)

    #Function to evaluate the class for a particular input points
    def make_prediction_point(self,point):
        '''
        This function will take an input point and calculate the
        probability of that point to belong in every classes.
        '''
        #Initializing the max probability and the prediction world
        max_world_prob=-1
        max_probable_world=None

        #Now iterating over all the class types to get their prob
        norm=0.0
        for world in class_vals:
            #Intiializing the prob of point with that of world
            prob=1*self.world_prob[world]
            #Now calculating the prob of these attributes in this world
            for attr in attr_vals.keys():
                #value of attributes in this point
                attr_val=point[attr]
                #Multiplying the attributes val prob in this world
                # print(world,attr,self.world_f_prob[world][attr].keys())
                prob*=self.world_f_prob[world][attr][attr_val]
            #Adding this probability for normalization
            norm+=prob
            #Saving the one with maximum prob
            if(prob>max_world_prob):
                max_world_prob=prob
                max_probable_world=world

        #Normalizing the prob before returning
        max_world_prob/=norm
        #Now returning the prediciton
        return max_probable_world,max_world_prob

    #Function to print the confusion matrix and accuracy
    def _print_evaluation_metrics(self,confusion_mat):
        '''
        This function will print the confusion matrix and the
        accuracy obtained on the given dataset
        '''
        #Initializing the counters for accuracy
        correct_count=0
        all_count=0

        #Now traversing over the matrix
        print("Print the confusion matrix")
        for label in class_vals:
            print("Actual Label: ",label)
            for pred in class_vals:
                count=confusion_mat[label][pred]
                print("\tcount:{}\t\tpred_label:{}".format(count,pred))
                #Adding up the correct count
                if(label==pred):
                    correct_count+=count
                #Adding the overall prediction
                all_count+=count
            print("####################")

        #Printing the accuracy
        print("Accuracy:\t",correct_count/all_count)


if __name__=="__main__":
    #Reading the dataset path
    filepath="dataset/nursery.data"
    infopath="dataset/nursery.c45-names"
    #Getting the dataset
    class_vals,attr_vals,train_df,valid_df=get_dataframe(filepath,
                                                            infopath)

    #Creating the classifier
    myNaive=gNaiveBayes(class_vals,
                        attr_vals,
                        train_df,
                        valid_df)
    #Calculating the world's probability
    myNaive.get_worlds_probability()
    myNaive.get_attr_prob_in_world()

    print("###############################################")
    print("Evaluating the results on traiing set")
    myNaive.evaluate_dataset(train_df)

    print("\n\n###############################################")
    print("Evaluating the results on validation set")
    myNaive.evaluate_dataset(valid_df)
