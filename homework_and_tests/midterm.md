# Midterm

## Preamble

You should submit a single `.py` with all questions answered.  Make sure the answer to each question appears in a function, a series of functions or a class.  The name of each function or class will be specified in the question.  Do not deviate from the name provided, otherwise you will get zero points for that question.

Question 1)

Write a function called `select_last_ten` that uses a sql query and the above `sql_query` function to get the last ten elements of a pandas dataframe.  The dataframe will be passed in.  Return the last ten elements.  Note: Order is not guaranteed in a sql query, so please account for this in the query.

Question 2)

Write a function called `min_max_pd` which takes in a dataframe and a column which must be in the dataframe.  It then returns the 95 percentile of the column times the 12 percentile of the column.  You must use pandas syntax.

Question 3)

Write a function called `aggregate_median` that takes in a dataframe and returns the median of each column in the dataframe.

Question 4) 

write a function called `modulo` that applies the modulo operator two variables.  The modulo operator in python looks like this `%`.

Question 5)

write a function called `is_odd` that checks if a variable is odd.

Question 6)

write a function called `counter_plus` that counts from start to finish where start and finish are passed in to the function.

Question 7)

write a function called `median_index` that returns the median index in a list.

Question 8) 

write a function that reverses a list called `reverse`

Question 9)

write a function called unique that takes in a list and returns it's unique elements, preserving the original order they appeared in the list.

Question 10)

write a function called `array_product` that takes in a list and returns the product of the numbers.

Question 11)

Write a function called `second_smallest_distance` that finds and returns the second smallest distance of all the adjacent numbers in a list.

Question 12)

write a function called `is_palindrome` that checks if a string is the same backwards and forwards.  Returns True if the string is an palindrome.

Question 13)

write a function called `count_palindrome` that counts the number of palindromes there are in a string.  Returns 0 if there are none.

Question 14)

write a function called `to_tensor` that takes a vector and a matrix and returns an order three tensor.

Question 15)

write a function called `mean_median_density` that slices the array between the mean and median.  The function returns the ratio of the length of the slice against the size of the full array.

Question 16)

write a function `is_linearly_independent` that takes in a matrix and returns True if the matrix is linearly independent.

Question 17)

write a function `multiply_vectors_py` that multiplies two vectors.  This function should not use any libraries.

Question 18)

write a function called `det_py` that takes the determinant of a list of lists of size 2 by 2.  This function should not use any libraries.

Question 19)

write a function called `read_data` which checks what kind of file extension you are using and calls the appropriate function and then returns a pandas dataframe.

Question 20)

write a function called `count_nulls` which returns the number of null values by column in a pandas dataframe.

