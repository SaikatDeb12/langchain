from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class Student(BaseModel):
    name: str
    age: Optional[int] = None
    email: EmailStr
    remarks: int = Field(gt=0, lt=100, default=40)


# student1 = Student(name="abc")
student1 = Student(name="abc", age=23, email="abc@mail.com", remarks=30)
print(student1)
# student1.model_dump_json()
