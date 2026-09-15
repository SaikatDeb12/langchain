from typing import TypedDict


class Person(TypedDict):
    name: str
    age: int


abc: Person = {"name": "abc", "age": 23}
print(abc)
