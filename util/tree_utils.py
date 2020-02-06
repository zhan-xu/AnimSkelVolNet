class Node(object):
    def __init__(self, name, pos):
        self.name = name
        self.pos = pos


class TreeNode(Node):
    def __init__(self, name, pos):
        super(TreeNode, self).__init__(name, pos)
        self.children = []
        self.parent = None
        self.order = None
        self.is_middle = False
