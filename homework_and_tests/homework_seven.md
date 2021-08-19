# Homework Seven

## Preamble

You should submit a single `.py` with all questions answered.  Make sure the answer to each question appears in a function, a series of functions or a class.  The name of each function or class will be specified in the question.  Do not deviate from the name provided, otherwise you will get zero points for that question.  Make sure to return at the end of all functions, never print.

one - normal
two - skew
three - exponential
four - multi-dim

## Question One

Write a function called read data that reads in `homework_seven.csv` as a pandas dataframe.  Name this dataframe `df`

## Questions 1, 2, 3

### Part One

Write a plotting function - 

Write a function that plots `df["question_one"]`, `df["question_two"]`, and `df["question_three"]` respectively as a histogram.

Hint: the sequence of bins will matter a lot.

### Part Two

Review your visualization for each question, what type of distribution do you believe this data to follow?  Come up with a formal hypothesis.

### Part Three

Try some central tendency functions and spread functions - which measures seem most appropriate for the given distribution?

Hint: you should plot the central tendency and some spread points to verify your hypothesis.

## Question 4

For this question, we'll need the columns `question_four_a` and `question_four_b`.

### Part One

Write a function to generate a scatter plot of the two columns.

Does it seem like the two columns are related?  Come up with a hypothesis as to how the two columns may be related.

### Part Two

Write a function to generate a joint plot of the two columns.

Does this seem to give you more evidence the two columns are related?  Or less?  Refine your hypothesis as appropriate.

### Part Three

Perform a correlation test on the two columns, does this conform with your hypothesis?

### Part Four

Write a function called covariance that calculates the covariance between two variables.  

Hint:  It's fine to use a library for this, but at least look up the formula so you understand what's happening.  

Bonus: Write this using just numpy.

### Part Five

Use your covariance function to calculate the covariance between the two variables.  

## Question 5

In this question we will return to the data in questions 1,2,3.

### Part One

For each column, `question_one`, `question_two`, `question_three` - 

Come up with a formal hypothesis for the distribution that each column follows.

### Part Two

For each column, `question_one`, `question_two`, `question_three` - 

Use the appropriate hypothesis test to verify your hypothesis.

### Part Three

Given that you now know the distributions of each of the columns, generate new sample data by first generating shape parameters and then using those shape parameters to generate new samples.  

### Part Four

Given the new sample data, run a hypothesis test to figure out how well the newly sampled data compares with your original dataset.


