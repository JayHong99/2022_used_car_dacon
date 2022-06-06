from tkinter import ON
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class preprocess : 
    def __init__(self, train : pd.DataFrame, test : pd.DataFrame
    ) -> pd.DataFrame :
        self.train = train
        self.test = test
        self.categorical = []

    def title(self) :  # 브랜드명만 가져와서 사용
        self.train['brand'] = self.train.title.apply(lambda x : x.split(' ')[0])
        self.test['brand']  = self.test.title.apply(lambda x : x.split(' ')[0])
        self.categorical.append(['title', 10])
        self.categorical.append(['brand', 10])
    
    def odometer(self) : 
        return
    
    def location(self) :
        self.categorical.append(['location',5])
        
    def isimported(self) : 
        self.categorical.append(['isimported',3])
    
    def enigne(self) : 
        self.categorical.append(['engine',4])
    
    def transmission(self) : 
        map_dict = {'automatic': 0, 'manual' : 1}
        self.train.transmission = self.train.transmission.map(map_dict)
        self.test.transmission = self.test.transmission.map(map_dict)
    
    def fuel(self) : 
        map_dict = {'petrol': 0, 'diesel' : 1}
        self.train.fuel = self.train.fuel.map(map_dict)
        self.test.fuel = self.test.fuel.map(map_dict)

    def paint(self) : 
        self.categorical.append(['paint',9])

    def year(self) : 
        self.train.year = self.train.year.apply(lambda x : x if (x < 1900 or x > 2022)  else -1)
        self.test.year = self.test.year.apply(lambda x : x if (x < 1900 or x > 2022)  else -1)
    
    def encoding(self, info) : 
        column, max_categories = info
        encoder = OneHotEncoder(max_categories=max_categories, handle_unknown='ignore')
        encoder.fit(self.train[[column]])
        columns = encoder.get_feature_names_out([column])

        train_oh = pd.DataFrame(
                                encoder.transform(self.train[[column]]).toarray(),
                                columns = columns
                                )
        self.train = pd.concat([self.train, train_oh], axis = 1).drop(columns = [column])
        test_oh = pd.DataFrame(
                                encoder.transform(self.test[[column]]).toarray(),
                                columns = columns
                                )
        self.test = pd.concat([self.test, test_oh], axis = 1).drop(columns = [column])


    def process(self) : 
        self.title()
        self.odometer()
        self.location()
        self.isimported()
        self.enigne()
        self.transmission()
        self.fuel()
        self.paint()
        self.year()

        for info in self.categorical : 
            self.encoding(info)

        self.train.drop(columns = ['id'], inplace = True)
        self.test.drop(columns = ['id'], inplace = True)
        return self.train, self.test