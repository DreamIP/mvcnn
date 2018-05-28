import sys
import os
import io

haddoc2_root =  'C:/Users/Kamel/dev/haddoc2/lib' # Path to your Haddoc2 Root
sys.path.insert(0, haddoc2_root)
import haddoc2
import quartus

if __name__ == '__main__':
    cwd = os.getcwd()
    cwd = '/'.join(cwd.split('\\'))
    proto_file = cwd + '/netpa3_s1_head.prototxt'
    model_file = cwd + '/../caffemodel/netpa3_s1.caffemodel'
    top_level_dir  = cwd + '/hdl_generated'
    bitWidth  = 7
    haddoc2.main(proto_file, model_file, top_level_dir, bitWidth)

    quartus_project_dir = 'quartus'
    haddoc2_hdl_lib = haddoc2_root + '/hdl'
    quartus.generateProject(haddoc2_hdl_lib = haddoc2_hdl_lib,
                            top_level_dir = top_level_dir,
                            out_dir = quartus_project_dir)
