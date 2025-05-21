import matplotlib.pyplot as plt
from math import atan2
# создаем файл с координатами, после всё можно менять через файл
with open('points.txt', 'w') as file:
    for point in [
        '-9.23 7.56',
        '2.87 -3.14',
        '8.11 1.92',
        '-4.55 -8.01',
        '5.69 9.47',
        '-1.33 4.28',
        '-6.88 -2.51',
        '7.02 -6.75',
        '3.94 5.83',
        '-0.11 -9.87',
        '9.52 -0.78',
        '-8.49 3.61',
        '4.76 -4.99',
        '-5.17 6.39',
        '0.89 8.64',
        '-3.06 -7.22',
        '6.21 0.53',
        '-7.65 -1.19',
        '1.58 -5.40',
        '-2.44 2.07'
    ]:
        file.write(point + '\n')

def readpoints():
  with open('/content/points.txt', 'r') as file:
    points = [[float(x), float(y)] for x, y in map(lambda line: line.split(), file)]
  return points
def lexmin(p):
    k = 0
    for i in range(1,len(p)):
        if p[i][0] < p[k][0]:
            k = i
        elif p[i][0] == p[k][0] and p[i][1] < p[k][1]:
            k = i
    return k

def relative_lexmin(k ,p):
    s = k + 1
    for i in range(k+2, len(p)):
        vec = [p[s][0] - p[k][0], p[s][1] - p[k][1]]
        vec2 = [p[i][0] - p[k][0], p[i][1] - p[k][1]]
        if (vec[0]*vec2[1] - vec[1]*vec2[0] < 0):
            s = i
    return s
def jarvis(p):
    k = lexmin(p)
    p0 = p[k]
    p[0], p[k] = p[k], p[0]
    k = relative_lexmin(0, p)
    p[1], p[k] = p[k], p[1]
    p.append(p[0])
    n = 1
    for i in range(1, len(p)):
        m = relative_lexmin(n, p)
        if p0 == p[m]:
            return n, p
        else:
            n+=1
            p[n], p[m] = p[m], p[n]
    return n, p
def sort_points(points):
    for i in range(len(points)):
        for j in range(i, len(points)):
            if (points[i][0]*points[j][1]-points[i][1]*points[j][0]>0):
                points[i], points[j] = points[j], points[i]
            elif (points[i][0]*points[j][1]-points[i][1]*points[j][0] ==0 and points[i][0]**2 +points[i][1]**2 < points[j][0]**2+ points[j][1]**2):
                points[i], points[j] = points[j], points[i]
def Graham(points):
    k=lexmin(points)
    p0 = points[k]
    points.remove(p0)
    for p in points:
        p[0] -=p0[0]
        p[1] -=p0[1]
    sort_points(points)#[p[0] - res[-1][0], p[1] - res[-1][1]])^(res[-1]-res[-2]
    res = [[0.,0.]]#[, p[1] - res[-1][1]]
    for p in points:
        while (len(res)>=2 and (p[0] - res[-1][0])*(res[-1][1]-res[-2][1]) - (p[1] - res[-1][1])*(res[-1][0]-res[-2][0])<=0):
            res.pop()
        res.append(p)
    for p in res:
        p[0] += p0[0]
        p[1] += p0[1]
    return res
