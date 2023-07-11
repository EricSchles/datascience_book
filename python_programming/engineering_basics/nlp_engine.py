import spacy

class NLPEngine:
    def __init__(self):
        self.spacy_nlp_sm = spacy.load("en_core_web_sm")
        self.spacy_nlp_md = None
        
    def get_tokens(self, doc: str) -> list:
        """
        Gets the tokens in the document
        
        Parameters:
        * doc - a string to process
        
        Returns:
        A list of tokens parsed from the text
        
        Examples:
        >>> nlp = NLPEngine()
        >>> nlp.get_tokens('Hello there friends')
        ['Hello', 'there', 'friends']
        """
        doc = self.spacy_nlp_sm(doc)
        return [token.text for token in doc]
    
    def get_pos(self, doc: str) -> list:
        """
        Gets the part of speech from 
        the tokens in the document.
        
        Parameters:
        * doc - a string to process
        
        Returns:
        A list of tuples of the form (token, part of speech)
        
        Examples:
        >>> nlp = NLPEngine()
        >>> nlp.get_pos('My dog spot runs fast!')
        [('My', 'DET'), ('dog', 'NOUN'), ('spot', 'NOUN'), ('runs', 'VERB'), ('fast', 'ADV'), ('!', 'PUNCT')]
        """
        doc = self.spacy_nlp_sm(doc)
        return [(token.text, token.pos_) for token in doc]
    
    def get_token_shape(self, doc: str) -> list:
        """
        Get the shape of all tokens in the document.
        
        The shape is the order of upper case and lower case
        letters per word.
        
        Parameters:
        * doc - the string to process
        
        Returns:
        A list of tuples of the form (token, shape)
        
        Examples:
        >>> nlp = NLPEngine()
        >>> nlp.get_token_shape('My dog spot runs fast')
        [('My', 'Xx'), ('dog', 'xxx'), ('spot', 'xxxx'), ('runs', 'xxxx'), ('fast', 'xxxx')]
        """
        doc = self.spacy_nlp_sm(doc)
        return [(token.text, token.shape_) for token in doc]
    
    def get_token_tags(self, doc: str) -> list:
        """
        Get the tags for each token.
        
        Paramters:
        * doc - the string to process
        
        Returns:
        A list of tuples of the form (token, tag)
        
        Examples:
        >>> nlp = NLPEngine()
        >>> nlp.get_token_tags('My dog spot runs fast')
        [('My', 'PRP$'), ('dog', 'NN'), ('spot', 'NN'), ('runs', 'VBZ'), ('fast', 'RB')]
        
        As you can see, this returns preposition, noun, noun, verb, adverb
        
        For a full list of tags see:
        https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        """
        doc = self.spacy_nlp_sm(doc)
        return [(token.text, token.tag_) for token in doc]
    
    def get_token_lemmas(self, doc: str) -> list:
        """
        Get the lemmas for each token.
        A lemma is normalized version of the token
        
        Parameters:
        * doc - the string to process
        
        Returns:
        A list of tuples of the form (token, lemma)
        
        Examples:
        >>> nlp = NLPEngine()
        >>> nlp.get_token_lemmas("My dog spot runs very fast.  He is the fastest doggy in the world")
        [('My', '-PRON-'), ('dog', 'dog'), ('spot', 'spot'), ('runs', 'run'), 
        ('very', 'very'), ('fast', 'fast'), ('.', '.'), (' ', ' '), 
        ('He', '-PRON-'), ('is', 'be'), ('the', 'the'), ('fastest', 'fast'), 
        ('doggy', 'doggy'), ('in', 'in'), ('the', 'the'), ('world', 'world')]
        
        What I think is important is the difference between 'fastest' and 'fast' here.  It illustrates how
        lemmas work - a lemma is sort of like the base word, and something like fastest is the word modified
        to make sense grammatically.  Usually nlp systems don't care much about grammar (in some cases).  
        So we can drop it.  So we can treat lemmatization as a sort of normalization of the word that throws
        out grammatical transforms.
        """
        doc = self.spacy_nlp_sm(doc)
        return [(token.text, token.lemma_) for token in doc]
    
    def get_token_ner_labels(self, doc:str) -> list:
        """
        Gets the named entity recognition labels for each token
        
        Parameters:
        * doc - the string to process
        
        Returns:
        A list of tuples of the form (token, ner_label)
        
        Examples:
        >>> nlp = NLPEngine()
        >>> nlp.get_token_ner_labels('I predict Google is going to buy Microsoft for one dollar')
        [('Google', 'ORG'), ('Microsoft', 'ORG'), ('one dollar', 'MONEY')]
        
        As you can see, the ner tagger knows that google and microsoft are organizations
        and that one dollar is money!
        """
        doc = self.spacy_nlp_sm(doc)
        return [(ent.text, ent.label_) for ent in doc.ents]
    
    def load_medium_language_model(self):
        """
        loads the medium language model into the
        spacy_nlp_md attribute which initially set to
        None.
        
        Parameters:
        * None
        
        Returns:
        Nothing
        """
        if not self.spacy_nlp_md:
            self.spacy_nlp_md = spacy.load("en_core_web_md")
        
    def how_similar(self, word_one: str, word_two: str) -> float:
        """
        A measure of similarity for two word vectors.
        The closer to 1.0 you get, the more similar
        the two words are.
        
        Similarity is calculated via the L2 norm:
        
        import math
        def L2_norm(first, second):
            differences = [second[index] - first[index]
                           for index in range(len(second))]
            squared_difference = [math.pow(diff, 2) 
                                  for diff in differences]
            sum_squared_difference = sum(squared_difference)
            return math.sqrt(sum_squared_difference)
        
        Note: A word vector is a compact matrix 
        representation of a word.  It encodes
        features about a word in an R^n space.
        
        Parameters:
        * word_one - a word vector
        * word_two - another word vector
        
        Returns:
        The L2 normed distance between two word vectors
        
        Examples:
        >>> nlp = NLPEngine()
        >>> nlp.how_similar('hamburger', 'hotdog')
        """
        self.load_medium_language_model()
        word_one = self.spacy_nlp_md(word_one)
        word_two = self.spacy_nlp_md(word_two)
        return word_one.similarity(word_two)
    
    def get_string_mode(self, tokens: list) -> str:
        """
        Gets the most frequently occurring string,
        known as the mode.
        
        Paramters:
        * tokens - a list of strings
        
        Returns:
        The most frequently occurring string in the
        list of strings.
        
        Examples:
        >>> nlp = NLPEngine()
        >>> nlp.get_string_mode(["en", "en", "fr"])
        'en'
        """
        unique_tokens = set(tokens)
        token_count = {}
        for token in unique_tokens:
            token_count[token] = tokens.count(token)
        return max(token_count)
    
    def language_detection(self, doc: str) -> str:
        """
        Returns the most likely language based on
        the language assigned to the most tokens.
        
        Because some words are defined across multiple
        languages, often times in english, french, and
        german as well as other languages with overlap,
        words will be true for more than one language.
        
        Therefore we take the most seen language per token.
        The likelihood of a tie should be very low, unless
        a set of text is in multiple languages.  Then using
        this method is inappropriate.
        
        Paramaters:
        * doc - the text to process
        
        Returns:
        The most likely language used in the text
        
        Examples:
        >>> nlp = NLPEngine()
        >>> nlp.language_detection(['Hello there friends'])
        'en'
        """
        doc = self.spacy_nlp_sm(doc)
        langs = [token.lang_ for token in doc]
        return self.get_string_mode(langs)
    
