import pandas as pd
import numpy as np
import time
import datetime

class RecordPeformance():
    
    def __init__(self,run_type = None):
        path = "performance_report/"
        self.clf_fn = "clf_performance.csv" 
        self.top_fn = "top_genes_performance.csv" 
        self.tot_fn = "total_performance.csv"

        if run_type.lower() == "test" or run_type.lower() == None:
            self.run_type = run_type
        else : 
            print ("\nError : Run Type Can Only be of type 'test' or None")
            raise ValueError
        if run_type.lower() == 'test':
            self.top_fn = 'test_'+self.top_fn 
            self.clf_fn = 'test_'+self.clf_fn
            self.tot_fn = 'test_'+self.tot_fn
        self.top_fn = path+self.top_fn 
        self.clf_fn = path+self.clf_fn
        self.tot_fn = path+self.tot_fn
            
    def top(self,top_gene_method : str,top_k_gene : int):
        self.top_gene_method = top_gene_method
        self.top_k_gene = top_k_gene
               
        self.top_columns = ['TimeStamp','Top_Gene_Method','Top_K_Genes','Time_Taken']          
        self.top_df = self.initializeCSV(self.top_fn,self.top_columns)
        self.top_time = datetime.datetime.now()

    def clf(self,clf_method : str,no_of_genes : int):
        self.clf_method = clf_method
        self.clf_no_of_gene = no_of_genes
              
        self.clf_columns = ['TimeStamp','Clf_Method','No_Of_Genes','Time_Taken',
        'Train_Acc','Test_Acc','Avg_Acc']
        self.clf_df = self.initializeCSV(self.clf_fn,self.clf_columns)
        self.clf_time = datetime.datetime.now()
        
    def tot(self,clf_method : str ,top_k_gene : int,top_gene_method: str):
        self.tot_clf_method = clf_method
        self.tot_top_k_gene = top_k_gene
        self.tot_top_gene_method = top_gene_method
       
        self.tot_columns = ['TimeStamp','Clf_Method','Top_K_Genes',
        'Top_Gene_Method','Time_Taken','Train_Acc','Test_Acc','Avg_Acc']
        self.tot_df = self.initializeCSV(self.tot_fn,self.tot_columns)
        self.tot_time = datetime.datetime.now()
        
    def top_start(self):
        self.top_time = datetime.datetime.now()
    def clf_start(self):
        self.clf_time = datetime.datetime.now()
    def tot_start(self):
        self.tot_time = datetime.datetime.now()

    def top_end(self):
        now = datetime.datetime.now()
        self.top_time_taken = (now - self.top_time)
        new_val_list = [
            (
                str(self.top_time),self.top_gene_method,self.top_k_gene,str(self.top_time_taken)
            )
        ]
        df_val = pd.DataFrame(np.array(new_val_list))
        df_val.columns = self.top_df.columns
        self.top_df = self.top_df.append(df_val)
        self.saveCSV(self.top_fn,self.top_df)

    def clf_end(self,train_acc : float,test_acc : float):
        now = datetime.datetime.now()
        self.clf_time_taken = (now - self.clf_time)
        new_val_list = [
            (
                self.clf_time,self.clf_method,self.clf_no_of_gene,self.clf_time_taken,
                train_acc,test_acc,(train_acc+test_acc)/2
            )
        ]
        df_val = pd.DataFrame(np.array(new_val_list))
        df_val.columns = self.clf_df.columns
        self.clf_df = self.clf_df.append(df_val)
        self.saveCSV(self.clf_fn,self.clf_df)

    def tot_end(self,train_acc : float,test_acc : float):
        now = datetime.datetime.now()
        self.tot_time_taken = (now - self.tot_time)
        new_val_list = [
            (
                self.tot_time,self.tot_clf_method,
                self.tot_top_k_gene,self.tot_top_gene_method,
                self.tot_time_taken,train_acc,test_acc,(train_acc+test_acc)/2
            )
        ]
        df_val = pd.DataFrame(np.array(new_val_list))
        df_val.columns = self.tot_df.columns
        self.tot_df = self.tot_df.append(df_val)
        self.saveCSV(self.tot_fn,self.tot_df)        


    def initializeCSV(self,file_name : str,columns: list) -> pd.DataFrame:
        
        try : 
            df =  pd.read_csv(file_name,sep=',')
            if list(df.columns) != columns :
                raise Exception
        except FileNotFoundError as e:
            df = pd.DataFrame(columns=columns)
        except Exception as e:
            print("!!! Exception Occured !!! \n",e)
        return df
        
    def saveCSV(self,file_name,df):
        df = df.set_index('TimeStamp',drop=True)
        df.to_csv(file_name, sep=',')

    def timeToDateTime(self,ts:time,datetime_format : str = '%Y-%m-%d %H:%M:%S') -> datetime:
        """Convert python's "time" to "datetime" String
        
        Arguments:
            ts {time} -- timestamp i.e. of "import time"
            datetime_format {str} -- fromat to return datetime (default: {'%Y-%m-%d %H:%M:%S'})
        Returns:
            datetime -- datetime of python i.e. of "import datetime"
        """
        return datetime.datetime.fromtimestamp(ts).strftime(datetime_format)
