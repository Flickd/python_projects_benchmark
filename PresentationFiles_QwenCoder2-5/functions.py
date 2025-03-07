def factorial(n, memo={}):
    # Check if the result for this value of n is already computed and stored in the memo dictionary
    if n in memo:
        return memo[n]

    # Base case: factorial of 0 or 1 is 1
    if n <= 1:
        return 1

    # Recursive calculation using memoization to store intermediate results
    memo[n] = n * factorial(n - 1, memo)
    return memo[n]


def nextfit(weight, c):
    # Initialize the number of bins required and remaining capacity of the current bin
    res = 0
    rem = c

    # Iterate through each weight in the list
    for _ in range(len(weight)):
        # Check if the current item can fit into the remaining space of the current bin
        if rem >= weight[_]:
            rem = rem - weight[_]
        else:
            # If not, increment the count of bins and reset the remaining capacity to the bin size
            res += 1
            rem = c - weight[_]

    return res


# Driver Code
weight = [2, 5, 4, 7, 1, 3, 8]
c = 10

print("Number of bins required in Next Fit :", nextfit(weight, c))
