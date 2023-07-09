import pydotplus
from IPython.display import Image
import numpy as np

file_name = '/root/final_project/metarl-offloading/env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1' \
            '/random.20.1.gv'
graph = pydotplus.graphviz.graph_from_dot_file(file_name)
# graph.write_png('test.png')
node_list = graph.get_node_list()
# jobs = [0] * len(node_list)

for _ in node_list:
    print(_.get_name())
    print(_.obj_dict['attributes']['size'])
    print(eval(_.obj_dict['attributes']['size']))
edge_list = graph.get_edge_list()
for _ in edge_list:
    print(_.get_source())
    print(_.get_destination())
