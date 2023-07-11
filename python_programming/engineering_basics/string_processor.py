class StringProcessor:
    """
    An object for processing strings.  The main methods of interest are:
    * to_camel_case
    * to_snake_case
    
    The preferred way to instantiate the class is as follows:
    >>> processor = StringProcessing()
    """
    def __init__(self):
        pass
    
    def remove_whitespace(self, name: str) -> str:
        """
        Removes whitespace between words

        Parameters:
        * name - the string which may or may not
        have whitespace.

        Returns:
        A string without whitespace between characters.

        Examples:
        >>> processor = StringProcessor()
        >>> processor.remove_whitespace("Hello There")
        'HelloThere'
        >>> processor.remove_whitespace("HelloThere")
        'HelloThere'
        """
        return "".join(name.split(" "))

    def get_upper_case_indices(self, name: str) -> list:
        """
        Gets the indices of all upper case words

        Parameters:
        * name - looks for uppercase 
        characters in this string

        Returns:
        A list of indices of the uppercase characters

        Examples:
        >>> processor = StringProcessor()
        >>> processor.get_upper_case_indices("HelloThere")
        [0, 5]
        >>> processor.get_upper_case_indices("HelloThereFriends")
        [0, 5, 10]
        """
        upper_case_indices = []
        for index, letter in enumerate(name):
            if letter.isupper():
                upper_case_indices.append(index)
        return upper_case_indices

    def get_lower_case_words(self, upper_case_indices: list, name: str) -> list:
        """
        Gets a list of the words, in lower case, split on uppercase
        characters

        Parameters:
        * upper_case_indices - a list of integers corresponding
        to upper case letters in the string
        * name - the string to split and process

        Returns:
        A list of words in lower case

        Examples:
        >>> processor = StringProcessor()
        >>> processor.get_lower_case_words([0, 5], "HelloThere")
        ['hello', 'there']
        >>> processor.get_lower_case_words([0, 5, 10], "HelloThereFriends")
        ['hello', 'there', 'friends']
        """
        start = 0
        lower_case_words = []
        for index in upper_case_indices[1:]:
            lower_case_words.append(
                name[start:index].lower()
            )
            start = index
        lower_case_words.append(
            name[index:].lower()
        )
        return lower_case_words

    def connect_words(self, lower_case_words: list) -> str:
        """
        Connects a list of words via a '_'

        Parameters:
        * lower_case_words - a list of lower case words

        Returns:
        A string of concatenated words, with '_' between
        each word.
        
        Examples:
        >>> processor = StringProcessor()
        >>> processor.connect_words(['hello', 'there'])
        'hello_there'
        >>> processor.connect_words(['hello', 'there', 'friends'])
        'hello_there_friends'
        """
        return "_".join(lower_case_words)

    def to_snake_case(self, name: str) -> str:
        """
        Takes a camel case string
        and makes it snake case

        Parameters:
        - name - the string to translate

        Returns:
        The snake cased string

        Example:
        >>> processor = StringProcessor()
        >>> processor.to_snake_case("HelloThere")
        'hello_there'
        >>> processor.to_snake_case("hello_there")
        'hello_there'
        >>> processor.to_snake_case("Hello There")
        'hello_there'
        """
        name = self.remove_whitespace(name)
        upper_case_indices = self.get_upper_case_indices(name)
        if upper_case_indices == []:
            return name
        lower_case_words = self.get_lower_case_words(
            upper_case_indices, name
        )
        return self.connect_words(lower_case_words)
    
    def split(self, name: str) -> list:
        """
        Split words on either "_" or " " 
        if present in name.
        
        Parameters:
        * name - the string to segment
        
        Returns:
        A tokenized list of words, separated
        by either "_" or whitespace
        
        Examples:
        >>> processor = StringProcessor()
        >>> processor.split("hello_there")
        ['hello', 'there']
        >>> processor.split("hello there")
        ['hello', 'there']
        >>> processor.split("hello there friends")
        ['hello', 'there', 'friends']
        """
        if "_" in name:
            return name.split("_")
        elif " " in name:
            return name.split(" ")
        else:
            return [name]
        
    def capitalize_words(self, words: list) -> list:
        """
        Takes in a list of words (strings) and
        capitalizes them.
        
        Parameters:
        * words - a list of words to captialize
        
        Returns:
        A list of words that are capitalized.
        
        Examples:
        >>> processor = StringProcessor()
        >>> processor.capitalize_words(['hello', 'there'])
        ['Hello', 'There']
        >>> processor.capitalize_words(['hello', 'there', 'friends'])
        ['Hello', 'There', 'Friends']
        """
        capitalized_words = []
        for word in words:
            if word:
                capitalized_words.append(
                    word.capitalize()
                )
        return capitalized_words
    
    def to_camel_case(self, name: str) -> str:
        """
        Takes a string of words, either
        separated by "_" or whitespace and
        returns a camel cased string
        
        Parameters:
        * name - the string to camel case
        
        Returns:
        A camel cased string, with no whitespace
        
        Examples:
        >>> processor = StringProcessor()
        >>> processor.to_camel_case("hello there")
        'HelloThere'
        >>> processor.to_camel_case('hello_there')
        'HelloThere'
        """
        words = self.split(name)
        words = self.capitalize_words(words)
        return "".join(words)
