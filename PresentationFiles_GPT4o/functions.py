def factorial(n, memo={}):
    """
    Calculate the factorial of a number using memoization.

    This function computes the factorial of a given number `n` using a
    recursive approach with memoization to optimize performance by storing
    previously computed results.

    Args:
        n (int): The number to compute the factorial of.
        memo (dict, optional): A dictionary to store previously computed
            factorials. Defaults to an empty dictionary.

    Returns:
        int: The factorial of the number `n`.
    """
    if n in memo:
        return memo[n]
    if n <= 1:
        return 1
    memo[n] = n * factorial(n - 1, memo)
    return memo[n]


def nextfit(weight, c):
    """
    Determine the number of bins required using the Next Fit algorithm.

    This function calculates the minimum number of bins required to
    accommodate a list of weights, where each bin has a capacity `c`.
    The Next Fit algorithm places each item into the current bin if it
    fits; otherwise, it starts a new bin.

    Args:
        weight (list of int): A list of weights to be placed into bins.
        c (int): The capacity of each bin.

    Returns:
        int: The number of bins required to fit all weights.
    """
    res = 0  # Initialize the number of bins used
    rem = c  # Initialize the remaining capacity of the current bin
    for _ in range(len(weight)):
        if rem >= weight[_]:
            rem = rem - weight[_]  # Place item in current bin
        else:
            res += 1  # Use a new bin
            rem = c - weight[_]  # Reset remaining capacity for the new bin
    return res


# Driver Code
weight = [2, 5, 4, 7, 1, 3, 8]
c = 10

print("Number of bins required in Next Fit :", nextfit(weight, c))
