'''
A parser of my skeleton file
Each row of skeleton file contains: hierarchical level, joint name, joint position (x, y, z), parent joint name
'''

from util.tree_utils import TreeNode

try:
    import Queue as Q  # ver. < 3.0
except ImportError:
    import queue as Q


class Skel:
    def __init__(self, filename=None):
        self.root = None
        if filename is not None:
            self.load(filename)

    def load(self, filename):
        with open(filename, 'r') as fin:
            lines = fin.readlines()
        for li in lines:
            words = li.split()
            if words[5] == "None":
                self.root = TreeNode(words[1], (float(words[2]), float(words[3]), float(words[4])))
                if len(words) == 7:
                    has_order = True
                    self.root.order = int(words[6])
                else:
                    has_order = False
                break
        self.loadSkel_recur(self.root, lines, has_order)

    def loadSkel_recur(self, node, lines, has_order):
        if has_order:
            ch_queue = Q.PriorityQueue()
            for li in lines:
                words = li.split()
                if words[5] == node.name:
                    ch_queue.put((int(li.split()[6]), li))
            while not ch_queue.empty():
                item = ch_queue.get()
                # print(item[0])
                li = item[1]
                ch_node = TreeNode(li.split()[1], (float(li.split()[2]), float(li.split()[3]), float(li.split()[4])))
                ch_node.order = int(li.split()[6])
                node.children.append(ch_node)
                ch_node.parent = node
                self.loadSkel_recur(ch_node, lines, has_order)
        else:
            for li in lines:
                words = li.split()
                if words[5] == node.name:
                    ch_node = TreeNode(words[1], (float(words[2]), float(words[3]), float(words[4])))
                    node.children.append(ch_node)
                    ch_node.parent = node
                    self.loadSkel_recur(ch_node, lines, has_order)

    def save(self, filename):
        fout = open(filename, 'w')
        this_level = [self.root]
        hier_level = 1
        while this_level:
            next_level = []
            for p_node in this_level:
                pos = p_node.pos
                parent = p_node.parent.name if p_node.parent is not None else 'None'
                if not p_node.order:
                    line = '{0} {1} {2:8f} {3:8f} {4:8f} {5}\n'.format(hier_level, p_node.name, pos[0], pos[1], pos[2], parent)
                else:
                    line = '{0} {1} {2:8f} {3:8f} {4:8f} {5} {6}\n'.format(hier_level, p_node.name, pos[0], pos[1], pos[2],
                                                                       parent, p_node.order)
                fout.write(line)
                for c_node in p_node.children:
                    next_level.append(c_node)
            this_level = next_level
            hier_level += 1
        fout.close()

    def get_joint_pos(self):
        joint_pos = {}
        this_level = [self.root]
        while this_level:
            next_level = []
            for node in this_level:
                joint_pos[node.name] = node.pos
                next_level += node.children
            this_level = next_level
        return joint_pos

    def normalize(self, scale, trans):
        this_level = [self.root]
        while this_level:
            next_level = []
            for node in this_level:
                node.pos /= scale
                node.pos = (node.pos[0] - trans[0, 0], node.pos[1] - trans[0, 1], node.pos[2] - trans[0, 2])
                for ch in node.children:
                    next_level.append(ch)
            this_level = next_level
