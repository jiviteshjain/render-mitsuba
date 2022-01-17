from render import Renderer

from argparse import ArgumentParser
import os

import numpy as np

XML_HEAD = """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="2,2,2" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        <sampler type="independent">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="1920"/>
            <integer name="height" value="1080"/>
            <rfilter type="gaussian"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
"""

XML_TAIL = """
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""

XML_SPHERE = """
    <shape type="sphere">
        <float name="radius" value="{radius}"/>
        <transform name="toWorld">
            <translate x="{x}" y="{y}" z="{z}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{r},{g},{b}"/>
        </bsdf>
    </shape>
"""

ROW_DIR = np.array([-3, 3, 0], dtype=np.float32)
ROW_DIR /= np.linalg.norm(ROW_DIR)
ROW_DIR = ROW_DIR[(1, 2, 0), ]
ROW_DIR[0] *= -1

COL_DIR = np.array([-3, -3, -3], dtype=np.float32)
COL_DIR /= np.linalg.norm(COL_DIR)
COL_DIR = COL_DIR[(1, 2, 0), ]
COL_DIR[0] *= -1

def run():
    parser = ArgumentParser()

    parser.add_argument('--self_color_files', type=str, nargs='+', default=[])
    
    parser.add_argument('--radius', type=float, default=0.015 * 1.6)

    parser.add_argument('--spacing', type=float, default=4)
    
    parser.add_argument('--mitsuba', type=str, default='/Users/jiviteshjain/Installs/mitsuba2/build/dist/mitsuba', help='Path to mitsuba2 installation.')
    parser.add_argument('--out_file', type=str, required=True, help='Output file path.')
    parser.add_argument('--in_dir', type=str, default='', help='Optional path prefex for input files.')
    
    args = parser.parse_args()

    renderer = Renderer(args.mitsuba, args.radius, args.radius, 1024, args.out_file, XML_HEAD, XML_TAIL, XML_SPHERE, scale_radius=True)
    
    for file in args.self_color_files:
        renderer.add_pcd(os.path.join(args.in_dir, file), 'self')

    renderer.preprocess_grid(ROW_DIR, COL_DIR, args.spacing)
    renderer.process()

if __name__ == '__main__':
    run()