# Homework Three

## Preamble

You should submit a single `.py` with all questions answered.  Make sure the answer to each question appears in a function, a series of functions or a class.  The name of each function or class will be specified in the question.  Do not deviate from the name provided, otherwise you will get zero points for that question. Make sure to return at the end of all functions, never print.

install the pandasql library.  You'll need the following code:

```python
from pandasql import sqldf
sql_query = lambda q: sqldf(q, locals())
```

## Question One

Part One)

Write a function called `generate_data` that generates a random pandas dataframe with N rows and the columns A, B, C.

Part Two)

Write a function called `select_first_ten` that uses a sql query and the above `sql_query` function to get the first ten elements of a pandas dataframe.  The dataframe will be passed in.  Return the first ten elements.  Note: Order is not guaranteed in a sql query, so please account for this in the query.

Part Three)

Write a function called `get_first_ten` which takes in a pandas dataframe and uses a slice in pandas to return the first ten elements of the dataframe.

## Question Two

Part One)

Write a function called `min_max_sql` which takes in a dataframe and a column which must be in the dataframe.  It then returns the maximum of the column minus the minimum of the column.  You must use a sql query to get the min and the max.

Part Two)

Write a function called `min_max_pd` which takes in a dataframe and a column which must be in the dataframe.  It then returns the maximum of the column times the minimum of the column.  You must use pandas syntax to query for the min and the max.

## Question Three

Part One)

write a function called `append_merge` that takes in two dataframes and combines appends them if all the columns are the same and merges them if only one column is the same.  Use the inner join if you are doing a merge.

Part Two)

Write a function called `aggregate_max` that takes in a dataframe and returns the max of each column in the dataframe.