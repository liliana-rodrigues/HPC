#import pvlib
import pandas as pd
import json 
import matplotlib.pyplot as plt

# Abra o arquivo JSON
with open('input_parameters.json') as file:
    data = json.load(file)

print(data['buy_price'])
