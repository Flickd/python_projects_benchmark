def add_numbers(a: int, b: int) -> int:
    return a + b

sum_result = add_numbers(10, 5)
print(f"Sum: {sum_result}")

class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
    
    def greet(self) -> str:
        return f"Hello, my name is {self.name} and I am {self.age} years old."

person1 = Person("Alice", 30)
print(person1.greet())

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = [x for x in numbers if x % 2 == 0]
print("Even numbers:", even_numbers)

data = {"name": "Bob", "age": 25, "city": "New York"}
print("Person's city:", data.get("city", "Unknown"))

try:
    result = 10 / 2
    print("Division result:", result)
except ZeroDivisionError:
    print("Cannot divide by zero")

with open("example.txt", "w") as file:
    file.write("This is an example file.")

with open("example.txt", "r") as file:
    content = file.read()
    print("File content:", content)

for index, value in enumerate(numbers):
    print(f"Index {index}: Value {value}")
