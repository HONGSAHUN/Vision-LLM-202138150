import numpy as np

np_arr = np.array([1, 2, 3, 4, 5])

# TODO: NumPy 연산 함수를 사용하여 배열을 정규화

np_arr_normalized = np_arr / np_arr.sum() # 어떤 식으로 정규화 할지 잘 모르겠어서 일단은 모든 값의 합이 1이 되도록 합으로 나누어보았습니다.
print(np_arr_normalized)

# TODO: NumPy 연산 함수를 사용하여 배열을 0.0 ~ 1.0 범위의 float32 배열로 변환
# TODO: 최종 배열의 최솟값 = 0.0, 최종 배열의 최댓값 = 1.0

np_arr_final = ((np_arr - np_arr.min()) / (np_arr.max() - np_arr.min())).astype(np.float32) # 1,2,3,4,5 배열에서 최솟값 1을 뺀 값에서 전체 범위 4로 나누기
print(np_arr_final)
print(np_arr_final.dtype)

print(f"최종 배열의 최솟값 = {np_arr_final.min()}, 최종 배열의 최댓값 = {np_arr_final.max()}\n")

raw_students = [
    "zg0505_짱구",
    "cs0822_철수",
    "mg0910_맹구",
    "yr0605_유리",
    "hi0205_훈이"
    ]

raw_scores = np.array([
    [95, 82, 88],   # 짱구
    [70, 65, 90],   # 철수
    [98, 100, 92],  # 맹구
    [60, 75, 80],   # 유리
    [81, 88, 84],   # 훈이
])

subjects = ["국어", "수학", "과학"]


# TODO: list comprehension을 활용해 raw_students 리스트에서 이름만 추출한 리스트 생성
student_names = [kname.split("_")[1] for kname in raw_students]
print("1. 학생 명단: ", end="")
for idx, name in enumerate(student_names):
    sep = ", " if idx < len(student_names) - 1 else "\n\n"
    print(name, end=sep)


# TODO: 전산 시스템에 입력된 모든 성적들을 1차원 배열로 출력 (총 15개)
all_scores = raw_scores.flatten()
print("2. 전산 시스템에 입력된 모든 성적: ", end="")
for idx, score in enumerate(all_scores):
    sep = ", " if idx < len(all_scores) - 1 else "\n\n"
    print(f"{score}점", end=sep)


# TODO: 전체 12개 성적 중 85점 이상 개수; 15개로 수정
over_85 = (raw_scores >= 85).sum()
print(f"3. 전체 15개 성적 중 85점 이상 개수: {over_85}개", end="\n\n")


# TODO: 학생별 평균 점수 출력
student_means = np.mean(raw_scores, axis=1)
student_means_int = [int(score) for score in student_means]

print("4. 학생별 평균 점수: ", end="")
for idx, (name, score) in enumerate(zip(student_names, student_means_int)):
    sep = ", " if idx < len(student_names) - 1 else "\n\n"
    print(f"{name} {score}점", end=sep) # 예상 출력: "4. 학생별 평균 점수: 짱구 00점, 철수 00점, 맹구 00점, 유리 00점, 훈이 00점"


# TODO: 평균 점수가 가장 높은 학생 선택
top_student_idx = np.array(student_means).argmax()
top_student_name = student_names[top_student_idx]
print(f"5. 최고 평균 수석 학생: {top_student_name} ({student_means[top_student_idx]:.2f}점)\n")  #예상 출력: "5. 최고 평균 수석 학생: XX (xx.xx점)"


# TODO: 전체 점수 기준, 최고 점수와 과목 그리고 해당 학생 계산 (modulo % 사용 추천)
overall_max_score = all_scores.max()
overall_max_idx = all_scores.argmax()
# if overall_max_idx % len(subjects) == 0:
#     overall_max_subject = "국어"
# elif overall_max_idx % len(subjects) == 1:
#     overall_max_subject = "수학"
# elif overall_max_idx % len(subjects) == 2:
#     overall_max_subject = "과학"
overall_max_subject = subjects[overall_max_idx % len(subjects)]
overall_max_student = student_names[overall_max_idx // len(subjects)]
print(f"6. 단일 과목 최고 점수: {overall_max_score}점 (과목: {overall_max_subject}, 학생: {overall_max_student})")  # 예상 출력: "6. 단일 과목 최고 점수: xx점 (과목: XX, 학생: XX)"
