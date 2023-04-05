
from dataclasses import dataclass
import os
from graphviz import Digraph

class bboxNode:
    def  __init__(self,idx,x1,y1,x2,y2, name=None,uuid=None):
        
        self.name = name
        self.uuid = uuid 
        if name is None:
            self.name = "node" + str(idx)
            self.uuid = "id" + str(idx)
        self.idx = idx
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.children = []
        
        
    def contains(self,node):
        return (self.x1 <= node.x1) and (self.y1 <= node.y1) and (self.x2 >= node.x2) and (self.y2 >= node.y2)
    
    def intersect(self,node):
        pointone = [self.x1, self.y1, self.x2, self.y2]
        pointtwo = [node.x1, node.y1, node.x2, node.y2]
        b = min(pointone[3], pointtwo[3])
        t = max(pointone[1], pointtwo[1])
        r = min(pointone[2], pointtwo[2])
        l = max(pointone[0], pointtwo[0])
        return (b >= t) and (r >= l)
        
    def insertInto(self, node):
        flag = False
        for child in self.children:
            if child.contains(node):
                child.insertInto(node)
                flag = True
                break
        if flag == False:
            self.children.append(node)
       
@dataclass
class bboxTree:

    root: bboxNode
    num: int = 0

    def __init__(self, x1, y1, x2, y2):
        self.root = bboxNode(-1, x1, y1, x2, y2, "start", "a")
        self.dot = Digraph()
        
        self.uuid =  ["a"]
        self.names = ["start"]
    
    def tailor(self,node):
        return bboxNode(node.idx, max(self.root.x1, node.x1),
                                 max(self.root.y1, node.y1),
                                 min(self.root.x2, node.x2),
                                 min(self.root.y2, node.y2), node.name, node.uuid)
        
    def insert(self, node:bboxNode):
        self.num += 1
        node = self.tailor(node)
        self.root.insertInto(node)
        self.uuid.append(node.uuid)
        self.names.append(node.name)
    
    def gen_graph_for_single_node(self, node, edges):

        if node.idx!=-1:
            for child in node.children:
                edges.append([node.idx, child.idx])
                edges.append([child.idx, node.idx])
        
        '''
        if len(node.children)>1:
            for j in range(1,len(node.children)):
                edges.append([node.children[j-1].idx,node.children[j].idx])
                edges.append([node.children[j].idx,node.children[j-1].idx])
            if len(node.children)>2:
                j = len(node.children)-1
                edges.append([node.children[0].idx,node.children[j].idx])
                edges.append([node.children[j].idx,node.children[0].idx])
        '''
        for i, node1 in enumerate(node.children):
            for j, node2 in enumerate(node.children):
                if i != j:
                    edges.append([node1.idx, node2.idx])
                
        for child in node.children:
            self.gen_graph_for_single_node(child, edges)
                    
    def gen_graph(self,edges=[]):
        self.gen_graph_for_single_node(self.root, edges)
        return edges
    
    def gen_tree_edges_single_node(self, node, edges):
        for child in node.children:
            edges.append([node.uuid, child.uuid])
        for child in node.children:
            self.gen_tree_edges_single_node(child, edges)
            
    def gen_tree_edges(self):
        edges = []
        self.gen_tree_edges_single_node(self.root, edges)
        return edges

    def visualize(self, visualization_path = None):
        edges = self.gen_tree_edges()
        for (uid, name) in zip(self.uuid, self.names):
            self.dot.node(uid, label = name)
        for edge in edges:
            self.dot.edge(edge[0], edge[1])
        dir,filename = os.path.split(visualization_path)
        print(visualization_path)
        self.dot.render(filename=visualization_path, view = False, format = 'pdf')

if __name__=='__main__':
    bbtree=bboxTree(0,0,375,375)
    bbtree.insert(bboxNode(1,2,4,45,35,name='test',uuid='x'))
    print(len(bbtree.root.children))
    bbtree.visualize("/media/sda1/ljz-workspace/code/uiGATv2/temp")
    '''
    dot = Digraph()
    dot.node('x','ss')
    dot.node('a','ss')
    dot.edge('x','a')
    dot.render(filename='temp',view=False,format='png')'''