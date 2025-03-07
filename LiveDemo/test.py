def factorial(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return 1
    memo[n] = n * factorial(n - 1, memo)
    return memo[n]


def nextfit(weight, c):
    res = 0
    rem = c
    for _ in range(len(weight)):
        if rem >= weight[_]:
            rem = rem - weight[_]
        else:
            res += 1
            rem = c - weight[_]
    return res


# Driver Code
weight = [2, 5, 4, 7, 1, 3, 8]
c = 10

print("Number of bins required in Next Fit :", nextfit(weight, c))
