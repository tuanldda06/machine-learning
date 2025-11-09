import math
from collections import Counter, defaultdict

# Dữ liệu mẫu
X = [
    {"Outlook": "Sunny",    "Wind": "Weak"},    # 0
    {"Outlook": "Sunny",    "Wind": "Strong"},  # 1
    {"Outlook": "Overcast", "Wind": "Weak"},    # 2
    {"Outlook": "Rain",     "Wind": "Weak"},    # 3
    {"Outlook": "Rain",     "Wind": "Strong"},  # 4
    {"Outlook": "Overcast", "Wind": "Strong"},  # 5
]
y = ["No", "No", "Yes", "Yes", "No", "Yes"]
attrs = ["Outlook", "Wind"]

class Node:
    def __init__(self, label=None, attr=None, children=None):
        self.label = label           # nếu là lá
        self.attr = attr             # thuộc tính dùng để tách tại nút này
        self.children = {} if children is None else children  # value -> node con

def entropy(labels):
    m = len(labels)
    if m == 0:
        return 0.0
    counts = Counter(labels)
    return -sum((c/m) * math.log2(c/m) for c in counts.values())

def info_gain(X, y, attr):
    m = len(y)
    ent_before = entropy(y)
    groups = defaultdict(list)                # value -> list index
    for i, row in enumerate(X):
        groups[row[attr]].append(i)
    ent_after = 0.0
    for idxs in groups.values():
        ys = [y[i] for i in idxs]
        ent_after += len(idxs) * entropy(ys) / m
    return ent_before - ent_after, groups

def most_common(y):
    return Counter(y).most_common(1)[0][0]

def id3(X, y, attrs):
    # dừng nếu thuần nhất
    if len(set(y)) == 1:
        return Node(label=y[0])
    # dừng nếu hết thuộc tính
    if not attrs:
        return Node(label=most_common(y))

    # chọn thuộc tính có information gain lớn nhất
    best_attr, best_gain, best_groups = None, -1.0, None
    for a in attrs:
        g, groups = info_gain(X, y, a)
        if g > best_gain:
            best_attr, best_gain, best_groups = a, g, groups

    # nếu không cải thiện thì trả về lá theo nhãn đa số
    if best_attr is None or best_gain <= 1e-12:
        return Node(label=most_common(y))

    # tạo nút quyết định và đệ quy cho từng nhánh
    node = Node(attr=best_attr)
    remaining = [a for a in attrs if a != best_attr]
    for val, idxs in best_groups.items():
        childX = [X[i] for i in idxs]
        childY = [y[i] for i in idxs]
        node.children[val] = id3(childX, childY, remaining)
    return node

def predict_one(node, x):
    while node.label is None:
        v = x.get(node.attr)
        if v not in node.children:   # giá trị lạ chưa gặp khi train
            return None
        node = node.children[v]
    return node.label

def predict(node, X):
    return [predict_one(node, x) for x in X]

# Demo nhỏ
if __name__ == "__main__":
    tree = id3(X, y, attrs)
    print("Dự đoán:", predict(tree, X))
