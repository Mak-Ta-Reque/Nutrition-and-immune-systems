import pandas as pd

predicted_output_file = "data/outputfiles_reduced/ideal_food_consumtion.csv"
df = pd.read_csv(predicted_output_file)
df.set_index('Nutrition Category', inplace=True)
print(df.loc['Protein_Supply_Quantity_Data'])
