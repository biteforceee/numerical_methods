import matplotlib as plt


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
class R:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __sub__(self, other):
        return R(self.x - other.x, self.y - other.y)

    def __XOR__(self, other):
        return self.x * other.y - self.y * other.x


def Graham(points):
    k=lexmin(points)
    p0 = points[k]
    points.remove(p0)
    for p in points:
        p[0] -=p0[0]
        p[1] -=p0[1]
    sort_points(points)#[p[0] - res[-1][0], p[1] - res[-1][1]])^(res[-1]-res[-2]
    res = [p0]#[, p[1] - res[-1][1]]
    for p in points:
        while (len(res)>=2 and (p[0] - res[-1][0])*(res[-1][1]-res[-2][1]) - (p[1] - res[-1][1])*(res[-1][0]-res[-2][0])<=0):
            res.pop()
        res.append(p)
    for p in res:
        p[0] += p0[0]
        p[1] += p0[1]
    return res


p = [[0.3,3],[0,0], [1,0], [1,1], [0,1], [0.5,0.5]]
n, a = jarvis(p)
b = Graham(p)
for i in range(0,n+2):
    print(a[i])
print("========")
for i in b:
    print(i)
'''
plt.subplot(3,3,1)
plt.xlim(-10,10)
plt.ylim(-10,10)

a , b = [], []
for i in range (len(p)):
    a[i] = p[i][0]
    b[i] = p[i][1]
    plt.scatter(a[i], b[i], marker='.', s=100, facecolors='red', edgecolors='red')

plt.show()

# Координаты точек
x = [0, 1, 2, 3]
y = [0, 1, 4, 9]

# Отображение точек
plt.scatter(x, y)

# Соединение точек линиями
plt.plot(x, y)

# Добавление осей и подписей
plt.xlabel('X')
plt.ylabel('Y')
plt.title('График зависимости Y от X')

# Показ графика
plt.show()'''