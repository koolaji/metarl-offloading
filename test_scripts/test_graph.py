import pydotplus
from IPython.display import Image
file_name = '/root/final_project/metarl-offloading/env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1' \
            '/random.20.1.gv'
graph = pydotplus.graphviz.graph_from_dot_file(file_name)
graph.write_png('test.png')

