# Author: Lê Công Đắt (MSSV: 20120454)
# Description: Individual Exercise 2 - Giải bài toán QHTT hình học
# Created: 20:50 07/03/2024
# Last modified: 21:05 18/03/2024

import math

import numpy as np


# Input:
# 3 // số ràng buộc
# 1 1 5 // ràng buộc 1, dạng ax+by <= c
# 0 1 2
# 1 2 6
# 30 50 // hàm mục tiêu F = 30x1+50x2
# Output:
# Danh sach cac diem cuc bien la:
# (0;0), (5;0), (4;1), (2;2), (0;2).
# 2) Mien rang buoc la bi chan.
# 3) GTLN la F = 190 tai x1 = 4, x2 = 1.
# GTNN là F = 0 tại x1 = 0, x2 = 0.

# Nhập các ràng buộc
def input_constraints(filename):
    with open(filename, 'r') as file:
        M = int(file.readline().strip())

        # Thêm 2 ràng buộc mặc định: x1 >= 0, x2 >= 0 <=> -x1 <= 0, -x2 <= 0
        a = [-1, 0]
        b = [0, -1]
        d = [0, 0]

        for _ in range(M):
            a_i, b_i, d_i = map(float, file.readline().split())
            a.append(a_i)
            b.append(b_i)
            d.append(d_i)
        M += 2
        c1, c2 = map(float, file.readline().split())
    return M, a, b, d, [c1, c2]


# Kiểm tra điểm có thỏa mãn ràng buộc không
def check_constraints(M, a, b, d, point):
    if point.x < 0 or point.y < 0:
        return False
    for i in range(M):
        if a[i] * point.x + b[i] * point.y - d[i] > EPS:
            return False
    return True


# Tìm các điểm cực biên
def extreme_points(M, a, b, d):
    extreme_point_list = []
    for i in range(0, M - 1):
        for j in range(i + 1, M):
            A = np.array([[a[i], b[i]], [a[j], b[j]]])
            B = np.array([d[i], d[j]])
            try:
                _point = np.linalg.inv(A).dot(B)
                point = Point(_point[0], _point[1])
                # print(point.x, point.y)
                if check_constraints(M, a, b, d, point) and not (extreme_point_list.__contains__(point)):
                    extreme_point_list.append(point)
            except np.linalg.LinAlgError:
                pass
    return extreme_point_list


# Kiểu điểm
class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return not self.__eq__(other)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y)


# Tích vô hướng của vector A và vector B
def dot(A, B):
    return A.x * B.x + A.y * B.y


EPS = 1e-9


# Góc giữa vector A và vector B
def calc_angle(A, B):
    if (A.length() * B.length()) < EPS:
        return 0
    return math.acos(dot(A, B) / A.length() / B.length())


# Trả về bao lồi với thứ tự các điểm được liệt kê cùng chiều kim đồng hồ
def convex_hull(p):
    n = len(p)
    if n <= 2:
        return p

    # Đưa điểm trái nhất lên đầu tập
    for i in range(1, n):
        if p[0].x > p[i].x:
            p[0], p[i] = p[i], p[0]

    # Tập bao lồi
    hull = [p[0]]

    # Dựng bao lồi
    while True:
        # Đỉnh cuối của tập hull
        P = hull[-1]

        # Đỉnh kế cuối của tập hull
        # Nếu hull.size() == 1 thì đặt đỉnh kế cuối là (P.x, P.y - 1)
        # Vì ban đầu hướng đang nhìn là từ dưới lên trên
        P0 = hull[-2] if len(hull) > 1 else Point(P.x, P.y - 1)

        # Q là đỉnh tiếp theo của tập hull
        Q = p[0]
        angle = calc_angle(P0 - P, Q - P)

        for i in range(1, n):
            if Q == P or Q == P0:
                Q = p[i]
                angle = calc_angle(P0 - P, Q - P)
                continue
            if p[i] == P or p[i] == P0:
                continue

            new_angle = calc_angle(P0 - P, p[i] - P)
            # Nếu góc (P0, P, Q) nhỏ hơn góc (P0, P, p[i]) thì gán Q = p[i]
            if abs(angle - new_angle) > EPS:
                if angle < new_angle:
                    Q = p[i]
                    angle = new_angle
            else:
                if (Q - P).length() > (p[i] - P).length():
                    Q = p[i]
                    angle = new_angle

        hull.append(Q)
        if hull[0] == hull[-1]:
            break

    # Đỉnh đầu tiên lặp lại ở cuối 1 lần
    hull.pop()

    return hull


def check_point_in_line(a, b, d, x, y):
    return abs(a * x + b * y - d) <= EPS


# Kiểm tra miền ràng buộc có bị chặn hay không
# True: Không bị chặn
# False: Bị chặn
def is_feasible_region_unbounded(M, a, b, d, convex_hull_points):
    # Nếu số điểm cực biên <= 2 thì miền ràng buộc không bị chặn
    if len(convex_hull_points) <= 2:
        return True
    for i in range(len(convex_hull_points) - 1):
        is_ok = True
        for j in range(M):
            # Xây dựng bao lồi của các điểm cực biên, duyệt qua các cạnh xem có cạnh nào
            # không thuộc một trong các đường đã cho hay không,
            # nếu có thì bao lồi đó bị “hở”, tức là sẽ không bị chặn.
            if (check_point_in_line(a[j], b[j], d[j], convex_hull_points[i].x,
                                    convex_hull_points[i].y) and check_point_in_line(a[j], b[j], d[j],
                                                                                     convex_hull_points[i + 1].x,
                                                                                     convex_hull_points[i + 1].y)):
                is_ok = False
                break
        if is_ok:
            return True

    is_ok = True
    for i in range(M):
        if (check_point_in_line(a[i], b[i], d[i], convex_hull_points[0].x, convex_hull_points[0].y) and
                check_point_in_line(a[i], b[i], d[i], convex_hull_points[-1].x, convex_hull_points[-1].y)):
            is_ok = False
            break
    return is_ok


def find_max_min(c, extreme_pts):
    max_val = -1e9
    min_val = 1e9
    max_point = []
    min_point = []
    for point in extreme_pts:
        val = c[0] * point.x + c[1] * point.y
        if val > max_val:
            max_val = val
            max_point = point
        if val < min_val:
            min_val = val
            min_point = point
    return max_val, max_point, min_val, min_point


def check_max_min_exist(M, a, b, c, d, x, y, max_val, min_val):
    if not check_constraints(M, a, b, d, Point(x, y)):
        return [True, True]
    val = c[0] * x + c[1] * y
    isMinExist = True
    isMaxExist = True
    if val > max_val:
        isMaxExist = False
    if val < min_val:
        isMinExist = False
    return [isMaxExist, isMinExist]


def main():
    M, a, b, d, c = input_constraints("input.txt")
    output_file = open("output.txt", "w", encoding="utf-8")

    # Danh sách các điểm cực biên
    print("\n1) Danh sách các điểm cực biên:")

    output_file.write("1) Danh sách các điểm cực biên:\n")
    extreme_pts = extreme_points(M, a, b, d)
    if len(extreme_pts) == 0:
        print("Không có điểm cực biên.")
        print("2) Miền ràng buộc không có")
        print("3) Không có giá trị lớn nhất và nhỏ nhất của hàm mục tiêu.")

        output_file.write("Không có điểm cực biên.\n")
        output_file.write("2) Miền ràng buộc không có\n")
        output_file.write("3) Không có giá trị lớn nhất và nhỏ nhất của hàm mục tiêu.\n")
        return
    for point in extreme_pts:
        print(f"({point.x};{point.y})", end=",")
        output_file.write(f"({point.x};{point.y}),")

    convex_hull_points = convex_hull(extreme_pts)

    # Kiểm tra miền ràng buộc có bị chặn hay không
    feasible_region_unbounded = is_feasible_region_unbounded(M, a, b, d, convex_hull_points)
    # Tìm giá trị lớn nhất và nhỏ nhất của hàm mục tiêu
    max_val, max_point, min_val, min_point = find_max_min(c, extreme_pts)
    if feasible_region_unbounded:
        print("\n2) Miền ràng buộc không bị chặn.")
        output_file.write("\n2) Miền ràng buộc không bị chặn.\n")
        x0 = 1e9
        y1 = 1e9
        [isMaxExist, isMinExist] = [True, True]
        for i in range(1, M):
            if b[i] != 0:
                y0 = (d[i] - a[i] * x0) / b[i]
                [temp1, temp2] = check_max_min_exist(M, a, b, c, d, x0, y0, max_val, min_val)
                isMaxExist = isMaxExist and temp1
                isMinExist = isMinExist and temp2
            if a[i] == 0:
                continue
            x1 = (d[i] - b[i] * y1) / a[i]
            [temp1, temp2] = check_max_min_exist(M, a, b, c, d, x1, y1, max_val, min_val)
            isMaxExist = isMaxExist and temp1
            isMinExist = isMinExist and temp2

        if isMaxExist and isMinExist:
            print(f"3) GTLN là F = {max_val} tại x1 = {max_point.x}, x2 = {max_point.y}.")
            print(f"GTNN là F = {min_val} tại x1 = {min_point.x}, x2 = {min_point.y}.")

            output_file.write(f"GTLN là F = {max_val} tại x1 = {max_point.x}, x2 = {max_point.y}.\n")
            output_file.write(f"GTNN là F = {min_val} tại x1 = {min_point.x}, x2 = {min_point.y}.\n")
        elif not isMaxExist and not isMinExist:
            print("3) Không có giá trị lớn nhất và nhỏ nhất của hàm mục tiêu.")

            output_file.write("3) Không có giá trị lớn nhất và nhỏ nhất của hàm mục tiêu.\n")
        elif not isMaxExist:
            print("3) GTLN không tồn tại")
            print(f"GTNN là F = {min_val} tại x1 = {min_point.x}, x2 = {min_point.y}.")

            output_file.write("3) GTLN không tồn tại\n")
            output_file.write(f"GTNN là F = {min_val} tại x1 = {min_point.x}, x2 = {min_point.y}.\n")
        elif not isMinExist:
            print("3) GTNN không tồn tại")
            print(f"3) GTLN là F = {max_val} tại x1 = {max_point.x}, x2 = {max_point.y}.")

            output_file.write("GTNN không tồn tại\n")
            output_file.write(f"GTLN là F = {max_val} tại x1 = {max_point.x}, x2 = {max_point.y}.\n")
    else:
        print("\n2) Miền ràng buộc là bị chặn.")
        print(f"3) GTLN là F = {max_val} tại x1 = {max_point.x}, x2 = {max_point.y}.")
        print(f"GTNN là F = {min_val} tại x1 = {min_point.x}, x2 = {min_point.y}.")

        output_file.write("\n2) Miền ràng buộc bị chặn.\n")
        output_file.write(f"3) GTLN là F = {max_val} tại x1 = {max_point.x}, x2 = {max_point.y}.\n")
        output_file.write(f"GTNN là F = {min_val} tại x1 = {min_point.x}, x2 = {min_point.y}.\n")


if __name__ == "__main__":
    main()
