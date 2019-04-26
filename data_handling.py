import pandas as pd
import numpy as np
import re
np.random.seed(1)

def get_dataframe(filepath,infopath):
    '''
    This function will read the csv file and generate a good old
    pandas dataframe for furthur analysis.
    '''
    #Reading the dataframe
    df=pd.read_csv(filepath,header=None)
    # print(df.dtypes)
    #Reading the attributes name and header
    class_vals,attrs,attr_vals=_read_attributes(infopath)

    #Setting up the header for the dataframe
    attrs.append("labels")
    df.columns=attrs
    print("\n\nPrinting the dataframe")
    #Shuffling the dataframe
    df=df.sample(frac=1,random_state=14).reset_index(drop=True)
    print(df.head())
    print("Shape of full dataset:",df.shape)

    #Now creating the train and test split
    split_pos=0.85
    train_df=df.loc[0:int(0.85*df.shape[0]),:]
    valid_df=df.loc[int(0.85*df.shape[0]):,:]

    #Printing the shape of each dataset (both feature and label)
    print("Shape of training set: ",train_df.shape)
    print("Shape of valid set   : ",valid_df.shape)

    return class_vals,attr_vals,train_df,valid_df


def _read_attributes(filepath):
    '''
    This function will read the attributes name present in the dataset
    as well as possible values of each attributes and the possible class
    in which to label the data point.
    '''
    #Initializing the attribute
    class_vals=[]
    attributes=[]
    attr_values={}

    with open(filepath,"r") as fhandle:
        #Reading the file line by line
        for line in fhandle:
            #Extracting the line with class labels
            if(":" in line):
                #Splitting the line according to pattern
                attr_vals=re.split("\:|\,|\.|\n| ",line)
                #Getting the attributes name and value
                attr=attr_vals[0]
                vals=[val for val in attr_vals[1:] if val !='']
                #Hashing the value
                attr_values[attr]=vals
                attributes.append(attr)
                print(attr,"\t: ",vals)
            elif("," in line):
                print("Printing the class labels")
                class_vals=re.split(",|\n| ",line)
                class_vals=[val for val in class_vals if val!='']
                print("classes\t: ",class_vals,"\n")
                print("Printing the attributes and their valus:")

    return class_vals,attributes,attr_values

if __name__=="__main__":
    #Reading the dataset path
    filepath="dataset/nursery.data"
    infopath="dataset/nursery.c45-names"

    get_dataframe(filepath,infopath)
