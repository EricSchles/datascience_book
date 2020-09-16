# Homework Three

## Preamble

You should submit a single `.py` with all questions answered.  Make sure the answer to each question appears in a function, a series of functions or a class.  The name of each function or class will be specified in the question.  Do not deviate from the name provided, otherwise you will get zero points for that question. Make sure to return at the end of all functions, never print.

## Question One

Part One)

write a function called `sum_elements_np` that takes in two numpy vectors and sums the elements.

write a function called `sum_elements_pd` that takes in two pandas series and sums the elements.

Part Two)

write a function called `row_selector` that selects the nth row of a pandas dataframe.

Part Three)

write a function called `index_selector` that selects the nth index of a pandas dataframe.

## Question Two

Part One)

write a function called `greater_than_median` that takes in a pandas series and returns all the elements greater than the median.

Part Two)

write a function called `dense_array` that slices the pandas series to all elements within one interquartile range of the median.

Part Three)

write a function called `mean_trimean_density` that slices the pandas series between the mean and trimean.  The function returns the ratio of the length of the slice against the size of the full array.