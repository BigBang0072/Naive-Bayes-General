import numpy as np
import pandas as pd
from data_handling import *
from collections import defaultdict

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
                #Now finding the probability of each of the attributes
                for fval,fframe in wframe.groupby(aval):
                    #Getting the count of this particular val of attr
                    feature_count=fframe.shape[0]
                    #Now saving the probability of this feature
                    prob=feature_count/world_count
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
