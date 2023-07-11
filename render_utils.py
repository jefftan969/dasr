import numpy as np
import os
import torch
import traceback
from cv2 import resize

from geom_utils import K2mat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OffscreenRenderer = None
SoftRendererAmbient = None
SoftRendererDirectional = None


# ===== Renderers

def make_pyrender_offscreen():
    if "pyrender" not in globals():
        global pyrender
        pyrender = import_pyrender()
    global OffscreenRenderer
    if OffscreenRenderer is None:
        OffscreenRenderer = pyrender.OffscreenRenderer(viewport_width=0, viewport_height=0, point_size=1.0)


def make_softras_ambient(near=1, far=10):
    if "softras" not in globals():
        global softras
        import soft_renderer as softras
    global SoftRendererAmbient
    if SoftRendererAmbient is None:
        SoftRendererAmbient = softras.SoftRenderer(
            image_size=0, sigma_val=1e-12, camera_mode="look_at", near=near, far=far,
            perspective=False, aggr_func_rgb="hard", light_mode="vertex",
            light_intensity_ambient=1, light_intensity_directionals=0, light_directions=[-1., -0.5, 1.]
        )


def make_softras_directional(near=1, far=10):
    if "softras" not in globals():
        global softras
        import soft_renderer as softras
    global SoftRendererDirectional
    if SoftRendererDirectional is None:
        SoftRendererDirectional = softras.SoftRenderer(
            image_size=0, sigma_val=1e-12, camera_mode="look_at", near=near, far=far,
            perspective=False, aggr_func_rgb="hard", light_mode="vertex",
            light_intensity_ambient=0, light_intensity_directionals=1, light_directions=[-1., -0.5, 1.]
        )


# ===== Pyrender Rendering

def import_pyrender(try_pyglet=False, try_osmesa=False, try_egl=True):
    if try_pyglet:
        try:
            import pyglet
            pyglet.window.Window().close() # Verify that Pyglet works
            import pyrender
            return pyrender
        except (pyglet.gl.ContextException, pyglet.canvas.xlib.NoSuchDisplayException):
            pass

    if try_osmesa:
        try:
            os.environ["PYOPENGL_PLATFORM"] = "osmesa"
            import pyrender
            return pyrender
        except ImportError:
            pass

    if try_egl:
        try:
            os.environ["PYOPENGL_PLATFORM"] = "egl"
            import pyrender
            return pyrender
        except ImportError:
            pass

    raise ImportError("Unable to import Pyrender")


# ===== SoftRas Rendering

def softras_render_mesh(
    vertices, faces, colors, camera_ks, *, directional=False, img_dim=480, scale=1
):
    """Render a deformed mesh using Softras

    Args
        vertices [..., n_points, 3]: Vertex positions of deformed mesh
        faces [..., n_faces, 3]: Faces of deformed mesh
        colors [..., n_points, 3]: Vertex colors of deformed mesh
        camera_ks [..., 6]: Camera intrinsics + width and height
        directional [bool]: Whether to add directional light
        img_dim [int]: Desired width and height of output image
        scale [float]: Scale factor to multiply vertices by prior to rendering

    Returns
        rendered [..., 4, H, W]: RGBA rendering of the given mesh
    """
    prefix_shape = vertices.shape[:-2]
    vertices = vertices.reshape(-1, vertices.shape[-2], 3) # T, npts, 3
    faces = faces.reshape(-1, faces.shape[-2], 3) # T, nfaces, 3
    colors = colors.reshape(-1, colors.shape[-2], 3) # T, npts, 3
    camera_ks = camera_ks.reshape(-1, camera_ks.shape[-1]) # T, 6

    # Convert camera intrinsics from image-pixel units to normalized "object" units
    # fx_n is in normalized unit, fx is in pixel unit (from banmo config)
    # We refer to camera_ks for values of img_w and img_h:
    #  - fx := fx_n * img_w / 2
    #  - fy := fy_n * img_h / 2
    #  - px := (px_n + 1) * img_w / 2
    #  - py := (py_n + 1) * img_h / 2
    # From banmo nnutils/banmo.py::render_dp()
    w = 2 * camera_ks[:, 4] # T,
    h = 2 * camera_ks[:, 5] # T,
    
    ks_scale = torch.stack([2 / w, 2 / h, 2 / w, 2 / h], dim=-1); del w, h # T, 4
    ks_offset = torch.tensor([[0, 0, -1, -1]], dtype=torch.float32, device=device) # 1, 4
    camera_ks = camera_ks[:, :4] * ks_scale + ks_offset; del ks_scale, ks_offset # T, 4
    
    Kmat = K2mat(camera_ks)[:, None, :, :]; del camera_ks # T, 1, 3, 3

    # Apply camera intrinsics transform (pre-multiply).
    # Then, divide x and y by z-coords to project onto image
    # Note Z shouldn't be negative here, otherwise image will flip
    vertices = torch.sum(Kmat * vertices[:, :, None, :], dim=-1); del Kmat # T, N, 3
    vertices[:, :, :2] /= (vertices[:, :, 2:] + 1e-6)

    # Import softras
    if directional:
        global SoftRendererDirectional
        if SoftRendererDirectional is None:
            make_softras_directional()
        SoftRenderer = SoftRendererDirectional
    else:
        global SoftRendererAmbient
        if SoftRendererAmbient is None:
            make_softras_ambient()
        SoftRenderer = SoftRendererAmbient

    # Create mesh
    offset = torch.tensor(SoftRenderer.transform.transformer._eye, device=vertices.device)[None, None]
    vertices_pre = (scale * vertices[:, :, :3]) - offset
    vertices_pre[:, :, 1] = -vertices_pre[:, :, 1]  # pre-flip
    mesh = softras.Mesh(vertices_pre, faces, textures=colors, texture_type="vertex")

    # Render mesh
    SoftRenderer.image_size = img_dim
    SoftRenderer.rasterizer.image_size = img_dim
    rendered = SoftRenderer.render_mesh(mesh) # T, 4, H, W
    rendered = rendered.view(prefix_shape + rendered.shape[-3:]) # ..., 4, H, W
    return rendered


def softras_render_points(
    vertices, colors, camera_ks, *, img_dim=224, scale=2, cube_radius=0.01, memory_limit=None
):
    """Render a set of points using Softras, by transforming each point into a cube

    Args
        vertices [bs, n_points, 3]: Vertex positions to render
        colors [bs, n_points, 3]: Vertex colors to render
        img_dim [int]: Desired width and height of output image
        scale [float]: Scale factor to multiply vertices by prior to rendering

    Returns
        rendered [bs, 4, H, W]: RGBA rendering of the given points
    """
    # Cube vertices and faces
    # Derived from cube.obj: https://gist.github.com/MaikKlein/0b6d6bb58772c13593d0a0add6004c1c
    cube_vertices = cube_radius * torch.tensor(
        [[1, -1, -1], [1, -1, 1], [-1, -1, 1], [-1, -1, -1], [1, 1, -1], [1, 1, 1], [-1, 1, 1], [-1, 1, -1]],
        dtype=torch.float32, device=device
    ) # 8, 3
    cube_faces = -1 + torch.tensor( # To turn 1-indexed into 0-indexed, we add -1
        [[2, 3, 4], [8, 7, 6], [5, 6, 2], [6, 7, 3], [3, 7, 8], [1, 4, 8],
         [1, 2, 4], [5, 8, 6], [1, 5, 2], [2, 6, 3], [4, 3, 8], [5, 1, 8]],
        dtype=torch.int64, device=device
    ) # 12, 3

    # Transform each point into a cube
    bs, n_points, _ = vertices.shape
    faces = torch.arange(n_points, dtype=torch.int64, device=device)[None].expand(bs, -1) # bs, n_points

    vertices_cubed = vertices[:, None, :, :] + cube_vertices[None, :, None, :] # bs, 8, n_points, 3
    faces_cubed = faces[:, None, :, None] * 8 + cube_faces[None, :, None, :] # bs, 12, n_points, 3
    colors_cubed = colors[:, None, :, :].repeat(1, 8, 1, 1) # bs, 8, n_points, 3

    vertices_cubed = vertices_cubed.view(bs, 8 * n_points, 3) # bs, 8*n_points, 3
    faces_cubed = faces_cubed.view(bs, 12 * n_points, 3) # bs, 12*n_points, 3
    colors_cubed = colors_cubed.view(bs, 8 * n_points, 3) # bs, 8*n_points, 3

    if memory_limit is None:
        chunk_size = bs
    else:
        memory_per_ch = 4 * img_dim * img_dim * 4
        chunk_size = (memory_limit + memory_per_ch - 1) // memory_per_ch

    rendered = []
    for i in range(0, bs, chunk_size):
        vertices_cubed_ch = vertices_cubed[i:i+chunk_size] # Nch, 8*n_points, 3
        faces_cubed_ch = faces_cubed[i:i+chunk_size] # Nch, 12*n_points, 3
        colors_cubed_ch = colors_cubed[i:i+chunk_size] # Nch, 8*n_points, 3
        camera_ks_ch = camera_ks[i:i+chunk_size] # Nch, 6

        rendered_ch = softras_render_mesh(
            vertices_cubed_ch, faces_cubed_ch, colors_cubed_ch, camera_ks_ch,
            texture_type="vertex", img_dim=img_dim, scale=scale
        ) # Nch, 4, H, W
        del vertices_cubed_ch, faces_cubed_ch, colors_cubed_ch
        rendered.append(rendered_ch)
        del rendered_ch

    del vertice_cubed, faces_cubed, colors_cubed
    rendered = torch.cat(rendered, dim=0) # bs, 4, H, W
    return rendered


# ===== PyRender Rendering

def pyrender_render_mesh(
    vertices, faces, colors, camera_ks, *, directional=False, img_dim=480, scale=1
):
    """Render a deformed mesh using Pyrender

    Args
        vertices [..., n_points, 3]: Vertex positions of deformed mesh
        faces [..., n_faces, 3]: Faces of deformed mesh
        colors [..., n_points, 3]: Vertex colors of deformed mesh
        camera_ks [..., 6]: Camera intrinsics + width and height
        directional [bool]: Whether to add directional light
        img_dim [int]: Desired width and height of output image
        scale [float]: Scale factor to multiply vertices by prior to rendering

    Returns
        rendered [..., 4, H, W]: RGBA rendering of the given mesh
    """
    prefix_shape = vertices.shape[:-2]
    vertices = vertices.reshape(-1, vertices.shape[-2], 3).cpu().numpy() # T, npts, 3
    faces = faces.reshape(-1, faces.shape[-2], 3).cpu().numpy() # T, nfaces, 3
    colors = colors.reshape(-1, colors.shape[-2], 3).cpu().numpy() # T, npts, 3
    camera_ks = camera_ks.reshape(-1, camera_ks.shape[-1]).cpu().numpy() # T, 6
    T = vertices.shape[0]

    # Pyrender converts camera intrinsics from image-pixel units to normalized "object" units:
    # https://github.com/mmatl/pyrender/blob/master/pyrender/camera.py#L396
    # fx_n is in normalized unit, fx is in pixel unit (from banmo config)
    # We refer to camera_ks for values of img_w and img_h:
    #  - fx := fx_n * img_w / 2
    #  - fy := fy_n * img_h / 2
    #  - px := (px_n + 1) * img_w / 2
    #  - py := (py_n + 1) * img_h / 2

    # Import pyrender
    global OffscreenRenderer
    if OffscreenRenderer is None:
        make_pyrender_offscreen()

    render_flags = pyrender.RenderFlags.SHADOWS_DIRECTIONAL
    render_flags |= pyrender.RenderFlags.OFFSCREEN

    # Scale fx, fy, px, py by `[w, h, w, h] / img_dim`
    w = 2 * camera_ks[:, 4] # T,
    h = 2 * camera_ks[:, 5] # T,
    camera_ks = camera_ks[:, :4] * img_dim / np.stack([w, h, w, h], axis=-1); del w, h # T, 4
    camera_pose = np.diag(np.array([1, -1, 1, 1], dtype=np.float32)) # 4, 4

    scene = None
    rendered = [] # T; 4, H, W
    for i in range(T):
        # Resize viewport
        OffscreenRenderer.viewport_width = img_dim
        OffscreenRenderer.viewport_height = img_dim

        if scene is None:
            # Create mesh and scene the first iteration
            primitive = pyrender.Primitive(
                positions=vertices[i], color_0=(255.0 * colors[i]).astype(np.uint8), indices=faces[i],
                mode=pyrender.GLTF.TRIANGLES
            )
            mesh = pyrender.Mesh([primitive])
            cam = pyrender.IntrinsicsCamera(
                fx=camera_ks[i, 0], fy=camera_ks[i, 1], cx=camera_ks[i, 2], cy=camera_ks[i, 3]
            )

            if directional:
                light = pyrender.DirectionalLight(color=np.full(3, 1, dtype=np.float32), intensity=10.0)
                scene = pyrender.Scene(ambient_light=np.full(3, 0.4, dtype=np.float32))
            else:
                light = pyrender.DirectionalLight(color=np.full(3, 1, dtype=np.float32), intensity=0.0)
                scene = pyrender.Scene(ambient_light=np.full(3, 1, dtype=np.float32))

            mesh_node = scene.add(mesh)
            light_node = scene.add(light, pose=camera_pose)
            cam_node = scene.add(cam, pose=camera_pose)

        else:
            # Update mesh and scene in future iterations
            primitive.positions = vertices[i]
            primitive.color_0 = colors[i]
            primitive.indices = faces[i]
            cam.fx = camera_ks[i, 0]
            cam.fy = camera_ks[i, 1]
            cam.cx = camera_ks[i, 2]
            cam.cy = camera_ks[i, 3]

        color, depth = OffscreenRenderer.render(scene, render_flags) # H, W, 3 | H, W
        scene = None  # TODO make more efficient by setting buffers without copy

        # Undo gamma correction and add alpha channel based on depth
        color = color / 255.0 # H, W, 3
        if not directional:
            color = np.where(color <= 0.0404482, color / 12.92, np.power((color + 0.055) / 1.055, 2.4)) # H, W, 3
        depth = np.where(depth == 0, depth, 1)[:, :, None] # H, W, 1
        color = np.moveaxis(np.concatenate([color, depth], axis=-1), -1, 0) # 4, H, W
        
        rendered.append(color)

    rendered = torch.tensor(np.stack(rendered, axis=0), dtype=torch.float32) # T, 4, H, W
    rendered = rendered.view(prefix_shape + rendered.shape[-3:]) # ..., 4, H, W
    return rendered
