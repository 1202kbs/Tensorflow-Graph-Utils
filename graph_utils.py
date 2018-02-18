import os

import tensorflow as tf
import numpy as np

from IPython.display import clear_output, Image, display, HTML


class Graph_Utils:


    def __init__(self, name, ckpt_dir):
        self.name = name
        self.ckpt_dir = ckpt_dir
        self.meta_dir = os.path.join(ckpt_dir, name + '.meta')

        self.sess, self.saver = self.__build()

        self.saver_def = self.saver.as_saver_def()

        self.graph = tf.get_default_graph()
        self.__graph_def = self.graph.as_graph_def()

        self.__nodes = list(self.__graph_def.node)
        self.__node_names = [node.name for node in self.__nodes]


    @property
    def graph_sess(self):
        self.__refresh()

        return self.graph, self.sess


    def __build(self):
        '''
        Build graph from given checkpoint and metagraph.
        '''
        sess = tf.InteractiveSession()
        saver = tf.train.import_meta_graph(self.meta_dir)
        saver.restore(sess, tf.train.latest_checkpoint(self.ckpt_dir))
        return sess, saver


    def get_tensors(self, node_names):
        '''
        Get the tensors of given node names.

        :param node_nmaes: a list of node names.
        :returns: a list of tensors.
        '''

        tensor_names = []

        for node_name in node_names:

            if ':0' not in node_name:

                tensor_names.append(node_name + ':0')

            else:

                tensor_names.append(node_name)

        res = []

        for tensor_name in tensor_names:

            res.append(self.graph.get_tensor_by_name(tensor_name))

        return res


    def __get_nodes(self, node_names):
        '''
        Get the node_defs of given node names.

        :param node_names: a list of node names.
        :returns: a list of node_defs.
        '''

        res = []

        for node_name in node_names:

            res.append(self.__nodes[self.__node_names.index(node_name)])

        return res


    def get_paths(self, start_node, end_node, checkpoints=[], fast_mode=False, visualize=True):
        '''
        Get a path / paths connecting two nodes.

        :param start_node: the beginning node of the path.
        :param end_node: the ending node of the path.
        :param checkpoints: a list of names of nodes the path must pass through.
        :param fast_mode: if True, returns a single path; else, returns all valid paths.
        :param visualize: if True, visualizes the path (only possible in the Jupyter Notebook environment).
        :returns: the path / paths connecting the given start and end node.
        '''

        self.__refresh()

        paths = self.__traverse(start_node, end_node, checkpoints, fast_mode)

        if visualize:
            self.__visualize_paths(paths)

        return paths


    def __traverse(self, start_node, end_node, checkpoints=[], fast_mode=False):
        '''
        Performs a recursive depth-wise path search.

        :param start_node: the beginning node of the path.
        :param end_node: the ending node of the path.
        :param checkpoints: a list of names of nodes the path must pass through.
        :param fast_mode: if True, returns a single path; else, returns all valid paths.
        :returns: the path / paths connecting the given start and end node.
        '''

        res = []

        if checkpoints:
            fast_mode = False

        if '^' in start_node:
            start_node = start_node[1:]

        if ':' in start_node:
            start_node = start_node.split(':')[0]

        candidates = list(self.__get_nodes([start_node])[0].input)

        if end_node in candidates:
            res.append(start_node)
            res.append(end_node)
            return res

        elif not candidates:
            return -1

        else:
            temp = []

            for candidate in candidates:
                traversed = self.__traverse(candidate, end_node, checkpoints, fast_mode)

                if traversed == -1:
                    continue

                else:
                    temp.append(traversed)

                    if fast_mode:
                        break

            if checkpoints:
                temp2 = []

                for i in range(len(checkpoints)):

                    temp3 = []
                    for j in range(len(temp)):

                        if checkpoints[i] in temp[j]:
                            temp3.append(temp[j])

                    if temp3:
                        temp2.extend(temp3)

                        for path in temp3:
                            temp.remove(path)

                if temp2:
                    temp = temp2

            if len(temp) == 1:
                temp = temp[0]

            if temp:
                res.append(start_node)

            else:
                return -1

            res.extend(temp)

            return res


    def __visualize_paths(self, paths=[]):
        '''
        Creates a new GraphDef object, attaches the given path / paths and then visualizes the GraphDef object. If no node names are given, visualizes the given Graph.
        
        :param graph: the Graph to visualize the paths.
        :param paths: list of node names indicating the path. List format must be same as that returned by __traverse() method.
        '''

        if not paths:
            self.visualize_graph()
        else:
            new_def = tf.GraphDef()
            self.__attach_paths(new_def, paths)
            self.__visualize(new_def)


    def __attach_paths(self, graph_def, paths):
        '''
        Recursively attaches the nodes indicated by paths to the given GraphDef object.

        :param graph_def: the GraphDef object to attach the nodes to.
        :param paths: list of node names indicating the path. List format must be same as that returned by __traverse() method.
        '''

        n = graph_def.node.add()
        n.MergeFrom(self.__get_nodes([paths[0]])[0])
        inputs = n.input

        while inputs:
            inputs.pop()

        if len(paths) == 1:
            return 0

        node = paths[1]

        if type(node) == list:

            for i in paths[1:]:

                inputs.append(i[0])
                self.__attach_paths(graph_def, i)

        else:

            inputs.append(node)
            self.__attach_paths(graph_def, paths[1:])


    def reroute(self, paths):
        '''
        Inserts nodes.

        :param paths: list indicating the path to insert nodes into. [[outputs, node_to_remove, node_to_insert], ...]
        '''

        self.__refresh()

        for path in paths:
        	outputs = self.__get_nodes(path[0])

        	for output in outputs:
        		output.input.remove(path[1])
        		output.input.append(path[2])

        self.__load_graph(self.__graph_def)


    def remove_nodes(self, node_names):
        '''
        Removes nodes.

        :param node_names: a list of node names indicating the nodes to remove.
        '''

        self.__refresh()

        nodes = self.__get_nodes(node_names)

        for node in nodes:
            self.__graph_def.node.remove(node)

        self.__load_graph(self.__graph_def)


    def remove_scopes(self, scope_names):
        '''
        Removes all nodes under given scopes.

        :param scope_names: a list of scope names to remove.
        '''

        self.__refresh()

        nodes = []
        for scope_name in scope_names:

            for node in self.__nodes:

                if scope_name in node.name:
                    nodes.append(node)

        for node in nodes:
            self.__graph_def.node.remove(node)

        self.__load_graph(self.__graph_def)


    def save(self, name, ckpt_dir):
        '''
        Saves the current graph.

        :param name: name of the graph.
        :param ckpt_dir: the directory to save the current graph.
        '''

        self.__refresh()
        save_path = os.path.join(ckpt_dir, name)
        self.saver.save(self.sess, save_path)


    def __load_graph(self, graph_def):
        '''
        Loads the given  GraphDef into the default Graph.

        :param sess: the Session in which the Variables of the given graph are initialized.
        :param graph: the Graph / GraphDef to load into the default Graph.
        :param output_node_names: a list containing the names of output nodes.
        :returns: a new InteractiveSession in which the Graph / GraphDef is loaded in. Same as calling tf.get_default_session().
        '''

        self.sess.close()

        tf.reset_default_graph()

        self.sess = tf.InteractiveSession()

        tf.import_graph_def(graph_def, name='')

        self.saver = tf.train.Saver(saver_def=self.saver_def)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.ckpt_dir))
        
        self.__refresh()


    def __refresh(self):
        '''
        Refresh class attributes.
        '''

        self.graph = tf.get_default_graph()
        self.__graph_def = self.graph.as_graph_def()
        self.__nodes = list(self.__graph_def.node)
        self.__node_names = [node.name for node in self.__nodes]


    def __strip_consts(self, graph_def, max_const_size=32):
        strip_def = tf.GraphDef()

        for n0 in graph_def.node:
            n = strip_def.node.add()
            n.MergeFrom(n0)
            
            if n.op == 'Const':
                tensor = n.attr['value'].tensor
                size = len(tensor.tensor_content)
                
                if size > max_const_size:
                    tensor.tensor_content = tf.compat.as_bytes("<stripped %d bytes>"%size)
        
        return strip_def


    def __rename_nodes(self, graph_def, rename_func):
        
        res_def = tf.GraphDef()

        for n0 in graph_def.node:
            n = res_def.node.add()
            n.MergeFrom(n0)
            n.name = rename_func(n.name)
            
            for i, s in enumerate(n.input):
                n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
        
        return res_def


    def __visualize(self, graph_def, max_const_size=32):
        '''
        Visualizes the given Graph / GraphDef.

        :param graph: the Graph / GraphDef to visualize. If not given, visualizes the default Graph.
        '''

        strip_def = self.__strip_consts(graph_def, max_const_size=max_const_size)

        code = """
            <script>
              function load() {{
                document.getElementById("{id}").pbtxt = {data};
              }}
            </script>
            <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
            <div style="height:600px">
              <tf-graph-basic id="{id}"></tf-graph-basic>
            </div>
        """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

        iframe = """
            <iframe seamless style="width:800px;height:620px;border:0" srcdoc="{}"></iframe>
        """.format(code.replace('"', '&quot;'))

        display(HTML(iframe))


    def visualize_graph(self):

        self.__refresh()
        self.__visualize(self.__graph_def)