# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 22:06:19 2022

@author: jocelyn


"""
#The goal of this project is to predict the amount of energy (electricity generation)
#Using GDP as an independent variable with the help of linear regression modeling
#our dependent variable is electricity generation
#our independent variable will be GDP
#The correlation of GDP with Electricity Generation is 99%

#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Something to consider is that this dataset is from 1 year ago so information is not that recent and
#may not align with more current data
#importing csv file recovered from Kaggle https://www.kaggle.com/datasets/pralabhpoudel/world-energy-consumption
WEC=pd.read_csv("C:/Users/jocelyn/Desktop/Python/WECproject/WorldEnergyConsumption.csv",index_col=0)
print(WEC)
df = pd.DataFrame(WEC)

#variables that we will explore and use for the model: country, year, electricity_generation
#gdp - GROSS DOMESTIC PRODUCT (Total real gross domestic product, inflation-adjusted) 

#We first did some exploration of the dataset (most of it is not shown here)

#We found that the correlation between Electricity generation and population is .93 - > 93%
print('Correlation between Electricity generation and population:')
print(WEC['electricity_generation'].corr(WEC['population']))

#gdp vs electricity generation
plt.scatter(WEC['gdp'],WEC['electricity_generation'])
plt.xlabel('GDP')
plt.ylabel('Energy Generation')
plt.title('Energy Generation compared to GDP')
plt.show()
#We find that the correlation between Electricity generation and 
#GDP is .99 - > 99%
print('Correlation between Electricity generation and GDP:')
print(WEC['electricity_generation'].corr(WEC['gdp']))

"""
creating new datasets where we can visualize better
"""
#only world information
world=WEC[(WEC['country']=='World')]
#Not considering world data as it is an outlier
WECnoW=WEC[(WEC['country']!='World')]

"""
****************************************************************************************************
Aqui es donde podemos cambiar el pais del que queremos predecir el consumo de energia basado en el GDP

Tambien se puede ajustar el modelo (como lo hacemos en la segunda parte del proyecto) para predecir 
el GDP basado en el año para algun pais en especifico

"""
#this will help us get data only after any specific year

WECyear=WECnoW[(WECnoW['year']>=1984)]
WECcountry=WECyear[(WECyear['country']=='Mexico')]


#informacion para GDP vs year, de 1900-1950 los ordenes de magnitud no nos ayudan a tener una buena grafica
WECyear2=WECnoW[(WECnoW['year']>=1950)]
#choosing an specific country to study
WECcountry2=WECyear2[(WECyear2['country']=='Mexico')]

# Para el calculo de GDP basado en los años, utilizamos informacion a partir de 1950 por ordenes de magnitud mas cercanos

"""
*****************************************************************************************************
"""


#Una primera visualizacion utilizando Seaborn para notar la relacion entre GDP Y generacion de energia
#esta es informacion solo considerando la data global
#ci es confidence interval
sns.regplot(x='gdp',
         y='electricity_generation',
         data=world,
         ci=0,
         scatter_kws={'alpha': 0.5})
plt.title('GDP vs Electricity Generation World')
plt.show()

#relacion GDP con generacion de electricidad para un country especifico (WEC country)
sns.scatterplot(x="gdp",
                y="electricity_generation",
                data=WECcountry
                )
sns.regplot(x="gdp",
         y="electricity_generation",
         data=WECcountry,
         ci=0,
         scatter_kws={'alpha': 0.5})
plt.title('GDP vs Electricity Generation Country')
plt.show()

"""

MODELO
#Regresion como indicador tecnico para predecir generacion de energia futura basado en gdp

"""


time2=np.arange(1,len(WECcountry)+1)
WECcountry['time2']=time2
WECcountry=(WECcountry[['time2','gdp','electricity_generation']]).dropna()
WECcountry
#So now, we have our data ordered so we can start our predictions and the model
reg2=np.polyfit(WECcountry['time2'],WECcountry['electricity_generation'],deg=1)
reg2
#Now, we will make an array to store the predicted values
trend2=np.polyval(reg2,WECcountry['time2'][-10:])
#standard deviation
std2=WECcountry['electricity_generation'][-10:].std()
#creating plot
plt.plot(WECcountry['time2'],WECcountry['electricity_generation'])
plt.plot(WECcountry['time2'][-10:],trend2,'r--')
#trend lines for Standard deviation
plt.plot(WECcountry['time2'][-10:],trend2 - std2,'g--')
plt.plot(WECcountry['time2'][-10:],trend2 + std2,'g--')
plt.xlabel('GDP')
plt.ylabel('Electricity generation')
plt.title('GDP vs Electricity generation')
plt.show()
#prediction about the future
predict = np.poly1d(reg2)
#Con este predict podemos predecir valores a futuro de la generacion de energia para el country que hayamos elegido 
#en la linea 67, 33 siendo el valor equivalente a 2017
predict(33)

#
#
#
#
#

#prediciendo GDP basado en los años

time=np.arange(1,len(WECcountry2)+1)
WECcountry2['time']=time
WECcountry2=(WECcountry2[['time','year','gdp']]).dropna()
WECcountry2
#So now, we have our data ordered so we can start our predictions and the model
reg=np.polyfit(WECcountry2['time'],WECcountry2['gdp'],deg=1)
reg
#Now, we will make an array to store the predicted values
trend=np.polyval(reg,WECcountry2['time'][-20:])
#standard deviation
std=WECcountry2['gdp'][-20:].std()
#creating plot
plt.plot(WECcountry2['time'],WECcountry2['gdp'])
plt.plot(WECcountry2['time'][-20:],trend,'r--')
#trend lines for Standard deviation
plt.plot(WECcountry2['time'][-20:],trend - std,'g--')
plt.plot(WECcountry2['time'][-20:],trend + std,'g--')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.title('Year vs GDP')
plt.show()

#prediction about the future
#Con este predict podemos predecir valores a futuro del gdp para el country que hayamos elegido 
#en la linea 73, 68 siendo el valor equivalente a 2017
predict2 = np.poly1d(reg)
predict2(68)
