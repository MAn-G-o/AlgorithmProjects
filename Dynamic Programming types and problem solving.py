################################## Dynamic Programming 

##################################### Max Profit with Fibonacci Series


def max_profit(prices):
    """
    This function takes a list of stock prices as input and returns the maximum profit that can be made by buying and selling one share of stock.
    """
    # Initialize variables
    max_profit = 0
    min_price = float('inf')
    
    # Iterate through the prices
    for price in prices:
        # Update the minimum price
        min_price = min(min_price, price)
        
        # Update the maximum profit
        max_profit = max(max_profit, price - min_price)
    
    return max_profit

# Driver code
prices = [7, 1, 5, 3, 6, 4]
profit = max_profit(prices)
print(f"Prices: {prices}")
print(f"Max Profit: {profit}")

##################### Longest Common Subsequence

def lcs(X, Y):
    """
    This function takes two strings as input and returns the length of the longest common subsequence (LCS) of the two strings.
    """
    # Initialize variables
    m = len(X)
    n = len(Y)
    L = [[0] * (n + 1) for i in range(m + 1)]
    
    # Build the LCS matrix
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    
    return L[m][n]

# Driver code
X = "AGGTAB"
Y = "GXTXAYB"
lcs_length = lcs(X, Y)
print(f"X: {X}")
print(f"Y: {Y}")
print(f"LCS Length: {lcs_length}")

############################################### Longest Increasing Subsequence

def lis(arr):
    """
    This function takes an array of numbers as input and returns the length of the longest increasing subsequence (LIS) of the array.
    """
    # Initialize variables
    n = len(arr)
    lis = [1] * n
    
    # Compute the LIS
    for i in range(1, n):
        for j in range(i):
            if arr[i] > arr[j] and lis[i] < lis[j] + 1:
                lis[i] = lis[j] + 1
    
    return max(lis)

# Driver code
arr = [10, 22, 9, 33, 21, 50, 41, 60]
lis_length = lis(arr)
print(f"Array: {arr}")
print(f"LIS Length: {lis_length}")

#################################### Knapsack solving Minimum cost problem 

def knapsack(weights, costs, capacity):
    n = len(weights)
    INF = float('inf')
    dp = [[INF for _ in range(sum(weights) + 1)] for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, sum(weights) + 1):
            if weights[i - 1] <= j:
                dp[i][j] = min(dp[i - 1][j], costs[i - 1] + dp[i - 1][j - weights[i - 1]])
            else:
                dp[i][j] = dp[i - 1][j]
    
    for j in range(sum(weights), -1, -1):
        if dp[n][j] <= capacity:
            return j

# Example usage
weights = [2, 3, 4, 5]
costs = [1, 2, 5, 6]
capacity = 8
print(knapsack(weights, costs, capacity)) # Expected output: 7

#################################### Knapsack solving MAXIMUM profit

def knapsack(weights, profits, capacity):
    n = len(weights)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for j in range(1, capacity + 1):
            if weights[i - 1] <= j:
                dp[i][j] = max(dp[i - 1][j], profits[i - 1] + dp[i - 1][j - weights[i - 1]])
            else:
                dp[i][j] = dp[i - 1][j]
    
    return dp[n][capacity]

# Example usage
weights = [2, 3, 4, 5]
profits = [3, 4, 8, 8]
capacity = 8
print(knapsack(weights, profits, capacity)) # Expected output: 15

