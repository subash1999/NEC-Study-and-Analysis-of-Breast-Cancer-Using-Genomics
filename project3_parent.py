import numpy as np
import pandas as pd
import os,sys
import joblib 
from sklearn import model_selection
import importlib, importlib.util



class Project3Parent():
    """This is the parent class for the all machine learning of project3 
    This contains the default path of gene data, 
    functions to manipulate the current_path,path and DataFrame
    
    ATTRIBUTE
    ---------
    --None--

    FUNCTION
    --------
    getCurrentPath() : 
        Return : str
    makeDF():
        Return : pandas.DataFrame
    getDF(): 
        Return : pandas.DataFrame
    getCsvPath():
        Return : str
    setCurrentPath() : 
        Parameters : str
    setDF(): 
        Parameters : pandas.DataFrame
    setCsvPath():
        Parameters : str        
    saveModelUsingJoblib():
        Parameters : model, filename_without_extension : str
        Return : bool
    getModelUsingJoblib():
        Parameters : filename_without_extension : str
        Return : model/False # model if "model" exists and "Flase" if the model doesnot exist
    getXYOfData():
        Return : tuple # (x,y) i.e. x is feature and y is output
    splitDataToTrainTest():
        Parameters : test_size_input: float
        Return : tuple # (x_train,x_test,y_train,y_test) 
    """

    def __init__(self,csv_path = 'Data/supervised_learning/output_dataset/gse_2034_processed_data.csv'):
        self.current_path = os.getcwd()
        # csv_path= "Data/supervised_learning/output_dataset/gse_2034_processed_data.csv"
        self.csv_path = csv_path
        self.df = None
        # making dataframe
        self.makeDF()
        self.getXYOfData()

    # getting the current path of command line
    def getCurrentPath(self) -> str: 
        """This returns the current path of the terminal
        
        Returns:
            String -- returns the current path of command line
        """
        return self.current_path

    # make the df from the csv_relative_path and return it
    def makeDF(self) -> pd.DataFrame :
        """This makes the data frame from the given csv_relative_path 
        
        Returns:
            pd.DataFrame -- dataframe of the genedata if file not found in 
            location this method will make df none ,pandas dataframe
        """
        try :
            self.df = pd.read_csv(self.csv_path,sep=',')
            self.getXYOfData()
        except Exception as E: 
            print("------------"*10)
            print("\t"*10,"***** FILE NOT FOUND *****")
            print("------------"*10)
            print("File Not Found... \nPlease provide a proper path using setCsvPath(csv_path:str) method")
            print("\t"*5,"-----"*7," Exception", "-----"*7,"\n",E)
            print("------------"*10)
            print("\t"*10,"***** FILE NOT FOUND END *****")
            print("------------"*10)
            self.df = None
        return self.df 
    
    # get the DataFrame of the geneData if present 
    def getDF(self) -> pd.DataFrame : 
        """This method returns dataframe of the data
        
        Returns:
            pd.DataFrame -- dataframe of the gene data if present and 
            if not present this returns none
        """
        return self.df

    # get csv relative path
    def getCsvPath(self) -> str:    
        """This function path to the csv file
        
        Returns:
            String -- returns the dataframe of the csv data
        """ 
        return self.csv_path
    
    # getting the current path of command line
    def setCurrentPath(self,current_path: str): 
        """Set the Current path of the terminal
        
        Prams:
            String -- set the current path of command line
        """
        self.current_path = current_path

    # make the df from the csv_relative_path
    def setDF(self,df: pd.DataFrame ):
        """Set the dataframe i.e 
        Prams: 
            DataFrame -- data frame of gene data
        """
        self.df = df        
    
    def returnDFfromNP(self, x : np.array, y : np.array ):
        """Set data frame from the np array of x and y
        
        Arguments:
            x {np.array} -- features in the np array
            y {np.array} -- results in the np array            
        """
        df_geo = self.getDF()['GEO_ACC']
        df_x = pd.DataFrame(x)
        df_y = pd.DataFrame(y[:,None])
        df_y.columns=["relapse"]
        df = pd.concat([df_x,df_y],axis=1)
        df = pd.concat([df_geo, df], axis = 1)
        return df

    # get csv path
    def setCsvPath(self,csv_path : str) -> str:    
        """This function sets path to the csv file
        Prams: 
            String -- set the current path of command line
        """ 
        self.csv_path = csv_path

    # save the model using joblib
    def saveModelUsingJoblib(self, model,filename_without_extension : str) -> bool :
        """This function saves "scikit learn" library's models using joblib
        
        Arguments:
            model {model_of_scikit_learn} -- One of the models of scikit learn
            filename_without_extension {str} -- the filename without extension should be mentioned here
        
        Returns:
            bool -- if successful return true and if unsuccessful false
        """
        ret_value = True

        try : 
            # Save the model as a pickle in a file 
            joblib.dump(model, filename_without_extension+'.pkl') 
        except Exception as e:
            print("-----"*20) 
            print("\t"*25,"\n******THE MODEL IS NOT SAVED (MODEL SAVE ERROR)******")
            print("-----"*20)
            print("\t"*20,"----"*5,"EXCEPTION MESSAGE","----"*5)
            print(e)
            print("-----"*20) 
            print("\t"*25,"\n******THE MODEL IS NOT SAVED END******")
            print("-----"*20)
            ret_value =  False

        return ret_value

    # save the model using joblib
    def getModelUsingJoblib(self, filename_without_extension : str) :
        """This function get "scikit learn" library's models which were saved using joblib
        
        Arguments:
            filename_without_extension {str} -- the filename without extension should be mentioned here
        
        Returns:
            model -- if successful return saved "MODEL" and if unsuccessful "False"
        """
        model = False
        
        try : 
            # Save the model as a pickle in a file 
            model = joblib.load(filename_without_extension+'.pkl')
        except Exception as e:
            print("-----"*20) 
            print("\t"*25,"\n******THE MODEL IS NOT LOADED (MODEL LOAD ERROR)******")
            print("-----"*20)
            print("\t"*20,"----"*5,"EXCEPTION MESSAGE","----"*5)
            print(e)
            print("-----"*20) 
            print("\t"*25,"\n******THE MODEL IS NOT LOADED END******")
            print("-----"*20)
            
        return model
    
    def getXYOfData(self,df = None) -> (np.array,np.array):
        """It returns the np array of the df in terms of features and result
       Arguments:
            df -- df of the data for project {(default: None)}, uses default df of class if None
       Returns:
            tuple -- tuple contains (x,y) of df for this project
        """
        if df is None:
            df = self.getDF()
        df = df.drop(columns= ['GEO_ACC'] )
        self.y = np.array(df['relapse'])
        self.x = np.array(df.drop(columns= ['relapse']))
        return (self.x,self.y)
    
    def splitDataToTrainTest(self,test_size_input: float = 0.2) -> (np.array,np.array,np.array,np.array):
        """It splits the data to train and test with the test_size
        
        Arguments:
            test_size_input {float} -- test_data size fraction (default = 0.2 : float)
           
        Returns:
            tuple -- tuple contains (x_train,x_test,y_train,y_test) of all np.array type
        """
        x,y = self.getXYOfData()
        self.x_train, self.x_test, self.y_train, self.y_test = model_selection.train_test_split(x,y,test_size=test_size_input)
        return (self.x_train, self.x_test, self.y_train, self.y_test)
   