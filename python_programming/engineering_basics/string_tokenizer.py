from string_processor import StringProcessor

class StringTokenizer(StringProcessor):
    def __init__(self):
        pass
    
    def tokenize(self, name: str, split_on: str ='') -> list:
        """
        'Tokenize' the string, meaning
        return a list of semantically viable symbols
        or words.  
        
        Parameters:
        * name - the string to tokenize
        
        Returns:
        A list of tokens (usually words) with all excess
        white space removed
        
        Examples:
        >>> string_tokenizer = StringTokenizer()
        >>> string_tokenizer.tokenize("Hello There My Friends")
        ['Hello', 'There', 'My', 'Friends']
        >>> string_tokenizer.tokenize("Hello-There-My-Friends", split_on='-')
        ['Hello', 'There', 'My', 'Friends']
        >>> string_tokenizer.tokenize("Hello  There \nMy \tFriends")
        ['Hello', 'There', 'My', 'Friends']
        >>> string_tokenizer.tokenize("Hello_There_My_Friends", split_on='_')
        ['Hello', 'There', 'My', 'Friends']
        """
        tokens = self.split(name, split_on=split_on)
        return [self.clean_endings(token) for token in tokens]
    
    def clean_endings(self, word: str) -> str:
        """
        Strips the endings off of words or characters
        
        Parameters:
        * name - the string to clean
        
        Returns:
        A string without excess space
        
        Examples:
        >>> string_tokenizer = StringTokenizer()
        >>> string_tokenizer.clean_endings("  Hello ")
        'Hello'
        >>> string_tokenizer.clean_endings("  \nHello\t ")
        'Hello'
        """
        word = word.lstrip()
        return word.rstrip()
        
    def split(self, name: str, split_on: str = '') -> list:
        """
        Splits a string into a list based on some typical
        cases.
        
        Parameters:
        * name - the string to split
        
        Returns:
        A list of strings, where each sub string
        is between the split character found.
        
        Examples:
        >>> string_tokenizer = StringTokenizer()
        >>> string_tokenizer.split("Hello there friends")
        ['Hello', 'there', 'friends']
        >>> string_tokenizer.split('Hello-there-friends', split_on='-')
        ['Hello', 'there', 'friends']
        """
        if split_on != '':
            return name.split(split_on)
        else:
            return name.split()
