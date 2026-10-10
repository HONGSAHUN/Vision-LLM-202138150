students = ["짱구", "철수", "맹구"]
scores = [85, 92, 78]

for index, (name, score) in enumerate(zip(students, scores), start = 1):
    if score >= 80:
        print(f"{1} {name} : {score}")