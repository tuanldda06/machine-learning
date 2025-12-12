import math
from collections import Counter, defaultdict

# Dữ liệu mẫu
X = [
    {"Outlook": "Sunny",    "Wind": "Weak"},    # row 0
    {"Outlook": "Sunny",    "Wind": "Strong"},  # row 1
    {"Outlook": "Overcast", "Wind": "Weak"},    # row 2
    {"Outlook": "Rain",     "Wind": "Weak"},    # row 3
    {"Outlook": "Rain",     "Wind": "Strong"},  # row 4
    {"Outlook": "Overcast", "Wind": "Strong"},  # row 5
]
y = ["No", "No", "Yes", "Yes", "No", "Yes"]

attrs = ["Outlook", "Wind"]

class Node:
    def __init__(self, label=None, attr=None, children=None):
        self.label = label           
        self.attr = attr             
        self.children =  children or {}  

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
    
    if not attrs:
        return Node(label=most_common(y))
    
    best_attr, best_gain, best_groups = None, -1.0, None
    for a in attrs:
        g, groups = info_gain(X, y, a)
        if g > best_gain:
            best_attr, best_gain, best_groups = a, g, groups

    
    if best_attr is None or best_gain <= 1e-12:
        return Node(label=most_common(y))

    
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
        if v not in node.children:   
            return None
        node = node.children[v]
    return node.label

def predict(node, X):
    return [predict_one(node, x) for x in X]


if __name__ == "__main__":
    tree = id3(X, y, attrs)
    print("Prediction:", predict(tree, X))
