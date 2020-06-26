from string_processor import StringProcessor

def test_remove_whitespace():
    processor = StringProcessor()
    assert 'HelloThere' == processor.remove_whitespace("Hello There")
    assert 'HelloThere' ==processor.remove_whitespace("HelloThere") 

def test_get_upper_case_indices():
    processor = StringProcessor()
    assert [0, 5] == processor.get_upper_case_indices("HelloThere")
    assert [0, 5, 10] == processor.get_upper_case_indices("HelloThereFriends")

def test_get_lower_case_words():
    processor = StringProcessor()
    assert ['hello', 'there'] == processor.get_lower_case_words([0, 5], "HelloThere")
    assert ['hello', 'there', 'friends'] == processor.get_lower_case_words([0, 5, 10], "HelloThereFriends")

def test_connect_words():
    processor = StringProcessor()
    assert 'hello_there' == processor.connect_words(['hello', 'there'])
    assert 'hello_there_friends' == processor.connect_words(['hello', 'there', 'friends'])
    
def test_to_snake_case():
    processor = StringProcessor()
    assert 'hello_there' == processor.to_snake_case("HelloThere")
    assert 'hello_there' == processor.to_snake_case("hello_there")
    assert 'hello_there' == processor.to_snake_case("Hello There")

def test_split():
    processor = StringProcessor()
    assert ['hello', 'there'] == processor.split("hello_there")
    assert ['hello', 'there'] == processor.split("hello there")
    assert ['hello', 'there', 'friends'] == processor.split("hello there friends")
    
def test_capitalize_words():
    processor = StringProcessor()
    assert ['Hello', 'There'] == processor.capitalize_words(['hello', 'there'])
    assert ['Hello', 'There', 'Friends'] == processor.capitalize_words(['hello', 'there', 'friends'])
    
def test_to_camel_case():
    processor = StringProcessor()
    assert 'HelloThere' == processor.to_camel_case("hello there")
    assert 'HelloThere' == processor.to_camel_case('hello_there')
    


