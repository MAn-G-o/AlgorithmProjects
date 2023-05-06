######################################## Brute force algorithm solving pattern matching problem 
 
def find_pattern(text, pattern):
    # Iterate over all possible starting indices of the pattern in the text
    for i in range(len(text) - len(pattern) + 1):
        # Initialize a flag to indicate if the pattern is found
        found = True
        
        # Iterate over all characters of the pattern
        for j in range(len(pattern)):
            # If the current character of the pattern does not match
            # the corresponding character of the text, set the flag to False
            if text[i+j] != pattern[j]:
                found = False
                break
        
        # If the pattern is found, return its starting index
        if found:
            return i
    
    # If the pattern is not found, return -1
    return -1

# Driver code to test the function
text = "hello world"
pattern = "world"
result = find_pattern(text, pattern)
print(f"The starting index of '{pattern}' in '{text}' is {result}.")

######################################################  Knuth-Morris-Pratt (KMP) Algorithm

def find_pattern_kmp(text, pattern):
    # Compute the failure function
    failure = [0] * len(pattern)
    i = 1
    j = 0
    while i < len(pattern):
        if pattern[i] == pattern[j]:
            failure[i] = j + 1
            i += 1
            j += 1
        elif j > 0:
            j = failure[j-1]
        else:
            i += 1
    
    # Search for the pattern in the text
    i = 0
    j = 0
    while i < len(text):
        if text[i] == pattern[j]:
            if j == len(pattern) - 1:
                return i - len(pattern) + 1
            i += 1
            j += 1
        elif j > 0:
            j = failure[j-1]
        else:
            i += 1
    
    # If the pattern is not found, return -1
    return -1

# Driver code to test the function
text = "hello world"
pattern = "world"
result = find_pattern_kmp(text, pattern)
print(f"The starting index of '{pattern}' in '{text}' is {result}.")

################################################## Rabin-Karp Algorithm

def find_pattern_rk(text, pattern):
    # Define the base and the modulus for the rolling hash function
    base = 256
    modulus = 101
    
    # Compute the hash of the pattern
    pattern_hash = 0
    for char in pattern:
        pattern_hash = (pattern_hash * base + ord(char)) % modulus
    
    # Compute the initial hash of the text
    text_hash = 0
    for char in text[:len(pattern)]:
        text_hash = (text_hash * base + ord(char)) % modulus
    
    # Iterate over all possible starting indices of the pattern in the text
    for i in range(len(text) - len(pattern) + 1):
        # If the hashes match, check if the strings match
        if text_hash == pattern_hash:
            if text[i:i+len(pattern)] == pattern:
                return i
        
        # Compute the next hash value
        if i < len(text) - len(pattern):
            text_hash = (text_hash * base - ord(text[i]) * pow(base, len(pattern), modulus) + ord(text[i+len(pattern)])) % modulus
    
    # If the pattern is not found, return -1
    return -1

# Driver code to test the function
text = "hello world"
pattern = "world"
result = find_pattern_rk(text, pattern)
print(f"The starting index of '{pattern}' in '{text}' is {result}.")

###################################################### Boyer-Moore Algorithm

def find_pattern_bm(text, pattern):
    # Compute the bad character shift table
    bad_char_shift = {}
    for i in range(len(pattern)):
        bad_char_shift[pattern[i]] = len(pattern) - i - 1
    
    # Iterate over all possible starting indices of the pattern in the text
    i = len(pattern) - 1
    while i < len(text):
        # Initialize a flag to indicate if the pattern is found
        found = True
        
        # Iterate over all characters of the pattern
        for j in range(len(pattern)):
            # If the current character of the pattern does not match
            # the corresponding character of the text, set the flag to False
            if text[i-j] != pattern[len(pattern)-j-1]:
                found = False
                
                # Compute the shift using the bad character rule
                shift = bad_char_shift.get(text[i-j], len(pattern))
                
                # Update the index and break out of the inner loop
                i += shift
                break
        
        # If the pattern is found, return its starting index
        if found:
            return i - len(pattern) + 1
    
    # If the pattern is not found, return -1
    return -1

# Driver code to test the function
text = "hello world"
pattern = "world"
result = find_pattern_bm(text, pattern)
print(f"The starting index of '{pattern}' in '{text}' is {result}.")