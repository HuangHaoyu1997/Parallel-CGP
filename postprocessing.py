"""
Postprocess the graph evolved by CGP.

- Simplify the obtained math formula.
- Visualize the expression tree corresponding to the formula that is embedded in the CGP graph.

Reference
1. [`geppy.support.simplification`](https://geppy.readthedocs.io/en/latest/geppy.support.html#module-geppy.support.simplification).
2. [Binary expression tree](https://en.wikipedia.org/wiki/Binary_expression_tree). Note that the operators are not required
to be binary though.

for Ubuntu
conda install graphviz
pip install pygraphviz

conda install graphviz
conda install pygraphviz
需要安装pygraphviz, https://www.lfd.uci.edu/~gohlke/pythonlibs/#pygraphviz
需要安装graphviz, https://graphviz.org/download/, 将路径下/bin/加入系统变量, 重启系统, 才不会报错"ValueError: Program neato not found in path."
"""
import cgp, pickle, operator, pygraphviz
import sympy as sp
from typing import Dict, Sequence
import networkx as nx
from configuration import config
from function import *



def extract_computational_subgraph(ind: cgp.Individual) -> nx.MultiDiGraph:
    """Extract a computational subgraph of the CGP graph `ind`, which only contains active nodes.
    Args:
        ind (cgp.Individual): an individual in CGP  
    Returns:
        nx.DiGraph: a acyclic directed graph denoting a computational graph
    See https://www.deepideas.net/deep-learning-from-scratch-i-computational-graphs/ and 
    http://www.cs.columbia.edu/~mcollins/ff2.pdf for knowledge of computational graphs.
    """
    # make sure that active nodes have been confirmed
    if not ind._active_determined:
        ind._determine_active_nodes()
        ind._active_determined = True
    # in the digraph, each node is identified by its index in `ind.nodes`
    # if node i depends on node j, then there is an edge j->i
    g = nx.MultiDiGraph()  # possibly duplicated edges
    for i, node in enumerate(ind.nodes):
        if node.active:
            f = ind.function_set[node.i_func]
            g.add_node(i, func=f.name)
            order = 1
            for j in range(f.arity):
                # print('name:',f.name)
                i_input = node.i_inputs[j] # 找父节点连接
                w = node.weights[j]
                g.add_edge(i_input, i, weight=w, order=order)
                order += 1
    return g


def simplify(g: nx.MultiDiGraph, input_names: Sequence = None, symbolic_function_map: Dict = None):
    """
    Compile computational graph `g` into a (possibly simplified) symbolic expression.
    将计算图g简化成一个符号表达式

    Args:
        g (nx.MultiDiGraph): a computational graph
        symbolic_function_map ([Dict], optional): Map each function to a symbolic one in `sympy`. Defaults to None.
            If `None`, then the `DEFAULT_SYMBOLIC_FUNCTION_MAP` is used.
        input_names (Sequence): a list of names, each for one input. If `None`, then a default name "vi" is used
            for the i-th input.
    
    Return:
        a (simplified) symbol expression
    
    For example, `add(sub(3, 3), x)` may be simplified to `x`. Note that this method is used to simplify the 
    **final** solution rather than during evolution. 

    用法：
    formula = simplify(g, ['v', 'h', 'g'])
    formula = round_expr(formula, config.PP_FORMULA_NUM_DIGITS)
    """
    if symbolic_function_map is None:
        symbolic_function_map = DEFAULT_SYMBOLIC_FUNCTION_MAP

    # toplogical sort such that i appears before j if there is an edge i->j, 【成环怎么办？】
    ts = list(nx.topological_sort(g))
    d = dict()
    for node_id in ts:
        if node_id < 0:  # input nodes in CGP
            d[node_id] = sp.Symbol(f"v{-node_id}" if input_names is None else input_names[-node_id - 1])
        else:  # a function node
            inputs = []
            # print(g.in_edges(node_id))
            for input_node_id in g.predecessors(node_id): # predecessors(n)得到n的父节点
                # possibly parallel edges
                for attr in g.get_edge_data(input_node_id, node_id).values():
                    inputs.append(
                        (input_node_id, attr["weight"], attr["order"]))
            inputs.sort(key=operator.itemgetter(2))

            args = (ip[1] * d[ip[0]] for ip in inputs)
            func = g.nodes[node_id]["func"]
            sym_func = symbolic_function_map[func]
            r = sym_func(*args)
            d[node_id] = sp.simplify(r) if config.PP_FORMULA_SIMPLIFICATION else r
    # the unique output is the last node
    return d[ts[-1]]


def round_expr(expr, num_digits):
    '''
    对符号表达式四舍五入
    '''
    # https://stackoverflow.com/questions/48491577/printing-the-output-rounded-to-3-decimals-in-sympy
    return expr.xreplace({n: round(n, num_digits) for n in expr.atoms(sp.Number)})


def visualize(g: nx.MultiDiGraph, to_file: str, input_names: Sequence = None, operator_map: Dict = None):
    """
    Visualize an acyclic graph `g`.无环图的可视化

    Args:
        g (nx.MultiDiGraph): a graph
        to_file (str): file path
        input_names (Sequence, optional): a list of names, each for one input. If `None`, then a default name "vi" is used
            for the i-th input. Defaults to None.
        operator_map (Dict, optional): Denote a function by an operator symbol for conciseness. Defaults to None. If `None`,
            then +-*/ are used.
    """

    from networkx.drawing.nx_agraph import to_agraph
    layout = 'dot' # 'neato' # 会改变graph的样式

    # label each function node with an operator
    if operator_map is None:
        operator_map = {operator.add.__name__: '+',
                        operator.sub.__name__: '-',
                        operator.neg.__name__: '-',
                        operator.mul.__name__: '*',
                        "protected_div": '/'}
    for n in g.nodes:
        attr = g.nodes[n]
        if n >= 0:  # function node
            if attr['func'] not in operator_map:
                print(f"Operator notation of '{attr['func']}'' is not available. The node id is shown instead.")
            
            # operator_map.get(attr['func'], n)返回的是该函数的哈希
            # attr['label'] = operator_map.get(attr['func'], n)

            # attr['func']是该函数的str名字
            attr['label'] = attr['func']
            # print(attr['func'], attr['label'], n)
            
            if g.out_degree(n) == 0:  # the unique output node
                attr['color'] = 'red' # 输出节点用红色表示
        
        else:  # 输入节点用绿色表示
            attr['color'] = 'green'
            attr['label'] = input_names[-n-1] if input_names is not None else f'v{-n}'

    ag: pygraphviz.agraph.AGraph = to_agraph(g)
    ag.layout(layout)
    ag.draw(to_file)
    
if __name__ == '__main__':
    with open('./results/CGP_LunarLanderContinuous-v2-79.pkl', 'rb') as f:
        pop = pickle.load(f)
    g = extract_computational_subgraph(pop[0])
    # formula = simplify(g, ) # ['x']
    # formula = round_expr(formula, config.PP_FORMULA_NUM_DIGITS)
    # print(pop[0].fitness, formula, type(formula.__str__))
    visualize(g, "./results/lunarlander.png", operator_map=DEFAULT_SYMBOLIC_FUNCTION_MAP)

