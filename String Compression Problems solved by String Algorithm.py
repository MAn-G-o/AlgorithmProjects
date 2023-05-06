######################################### Brute force String compression problem solving

def compress_string(s):
    # Initialize the compressed string
    compressed = ""
    
    # Initialize the count of the current character
    count = 1
    
    # Iterate over all characters of the string
    for i in range(1, len(s)):
        # If the current character is different from the previous one
        if s[i] != s[i-1]:
            # Append the previous character and its count to the compressed string
            compressed += s[i-1] + str(count)
            
            # Reset the count to 1
            count = 1
        else:
            # Increment the count
            count += 1
    
    # Append the last character and its count to the compressed string
    compressed += s[-1] + str(count)
    
    # Return the compressed string if it is shorter than the original string,
    # otherwise return the original string
    return compressed if len(compressed) < len(s) else s

# Driver code to test the function
s = "aabcccccaaa"
result = compress_string(s)
print(f"The compressed version of '{s}' is '{result}'.")

############################################# Knuth-Morris-Pratt (KMP) Algorithm

def kmp_search(pattern, text):
    """
    This function uses the Knuth-Morris-Pratt (KMP) algorithm to search for a pattern within a text.
    It returns the starting index of the first occurrence of the pattern in the text, or -1 if the pattern is not found.
    """
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
            j = failure[j - 1]
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
            j = failure[j - 1]
        else:
            i += 1

    return -1

def compress_string(text):
    """
    This function compresses a string using the following method:
    For each character in the string, it finds the longest substring starting from that character that repeats at least once.
    It then replaces that substring with its length followed by the substring itself.
    If no such substring is found, it simply appends the character to the result.
    """
    result = ""
    i = 0
    while i < len(text):
        # Find the longest repeating substring starting from position i
        length = 0
        for j in range(i + 1, len(text)):
            pattern = text[i:j]
            index = kmp_search(pattern, text[j:])
            if index != -1:
                length = len(pattern)
            else:
                break

        # Append the compressed substring or character to the result
        if length > 0:
            result += str(length) + text[i:i+length]
            i += length
        else:
            result += text[i]
            i += 1

    return result

# Driver code to test the above functions
text = "abababababab"
compressed_text = compress_string(text)
print(f"Original text: {text}")
print(f"Compressed text: {compressed_text}")


######################## Rabin-Karp algorithm

def rabin_karp_search(pattern, text):
    """
    This function uses the Rabin-Karp algorithm to search for a pattern within a text.
    It returns the starting index of the first occurrence of the pattern in the text, or -1 if the pattern is not found.
    """
    # Define constants for the rolling hash function
    d = 256  # Number of characters in the input alphabet
    q = 101  # A prime number

    # Compute the length of the pattern and text
    m = len(pattern)
    n = len(text)

    # Compute the hash value of the pattern and the first window of text
    p = 0  # Hash value for pattern
    t = 0  # Hash value for text
    h = pow(d, m - 1) % q

    for i in range(m):
        p = (d * p + ord(pattern[i])) % q
        t = (d * t + ord(text[i])) % q

    # Slide the pattern over text one by one
    for i in range(n - m + 1):
        # Check if the hash values of the current window of text and pattern match
        if p == t:
            # Check if all characters of the current window of text and pattern match
            match = True
            for j in range(m):
                if text[i + j] != pattern[j]:
                    match = False
                    break

            if match:
                return i

        # Compute the hash value for the next window of text
        if i < n - m:
            t = (d * (t - ord(text[i]) * h) + ord(text[i + m])) % q

            # We might get negative value of t, converting it to positive
            if t < 0:
                t += q

    return -1

def compress_string(text):
    """
    This function compresses a string using the following method:
    For each character in the string, it finds the longest substring starting from that character that repeats at least once.
    It then replaces that substring with its length followed by the substring itself.
    If no such substring is found, it simply appends the character to the result.
    """
    result = ""
    i = 0
    while i < len(text):
        # Find the longest repeating substring starting from position i
        length = 0
        for j in range(i + 1, len(text)):
            pattern = text[i:j]
            index = rabin_karp_search(pattern, text[j:])
            if index != -1:
                length = len(pattern)
            else:
                break

        # Append the compressed substring or character to the result
        if length > 0:
            result += str(length) + text[i:i+length]
            i += length
        else:
            result += text[i]
            i += 1

    return result

# Driver code to test the above functions
text = "abababababab"
compressed_text = compress_string(text)
print(f"Original text: {text}")
print(f"Compressed text: {compressed_text}")

############################################### Boyer-Moore Algorithm for string compression

def boyer_moore_compression(text):
    """
    This function takes a string as input and returns a compressed version of the string using the Boyer-Moore algorithm.
    """
    # Initialize variables
    compressed_text = ""
    i = 0
    
    # Iterate through the text
    while i < len(text):
        # Find the next occurrence of the current character
        j = text.find(text[i], i + 1)
        
        # If the character is not found, add it to the compressed text and move to the next character
        if j == -1:
            compressed_text += text[i]
            i += 1
        else:
            # If the character is found, add the number of occurrences and the character to the compressed text
            compressed_text += str(j - i) + text[i]
            i = j
    
    return compressed_text

# Driver code
text = "aaabbbcccdddeeefff"
compressed_text = boyer_moore_compression(text)
print(f"Original text: {text}")
print(f"Compressed text: {compressed_text}")