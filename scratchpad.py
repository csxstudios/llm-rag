import os
import helpers

file = os.path.basename("data/oscars.csv")
print(file)
file_name, file_extension = os.path.splitext(file)
print(file_name)

helpers.csv_to_sql_db("data/oscars.csv")