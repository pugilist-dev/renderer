import os
import numpy as np
import trimesh
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import random
from PIL import Image

class ObjectRenderer:
    def __init__(self, data_path="./data"):
        """Initialize the renderer with the path to data folder"""
        self.data_path = data_path
        self.window_size = (800, 600)
        self.fov = 45
        pygame.init()
        pygame.display.set_mode(self.window_size, DOUBLEBUF | OPENGL)
        
    def setup_camera(self):
        """Setup the camera with perspective projection"""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, (self.window_size[0]/self.window_size[1]), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        
        # Enable depth testing and lighting
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        
        # Set light position and properties
        glLightfv(GL_LIGHT0, GL_POSITION, (0, 0, 10, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))

    def random_viewpoint(self):
        """Generate random camera position on a sphere"""
        theta = random.uniform(0, 2 * np.pi)
        phi = random.uniform(0, np.pi)
        radius = 3.0  # Reduced radius to see object better
        
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        
        return x, y, z
    
    def render_object(self, obj_path):
        """Render a single textured object and save the image"""
        # Load the mesh with texture and materials
        mesh = trimesh.load(obj_path, force='mesh', process=False)

        if not isinstance(mesh, trimesh.Trimesh):
            print(f"Skipping non-trimesh object: {obj_path}")
            return

        print(f"Mesh loaded: vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")

        # Center and scale the mesh
        mesh.vertices -= mesh.center_mass
        scale = 1.0 / max(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
        mesh.vertices *= scale

        # Load texture if available
        texture_id = None
        if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'image') and mesh.visual.material.image is not None:
            image = mesh.visual.material.image
            image_data = image.transpose(Image.FLIP_TOP_BOTTOM).convert('RGBA').tobytes()
            width, height = image.size

            # Generate OpenGL texture
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)

            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glEnable(GL_TEXTURE_2D)

        # Clear buffers
        glClearColor(0.5, 0.5, 0.5, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Set random camera viewpoint
        x, y, z = self.random_viewpoint()
        gluLookAt(x, y, z, 0, 0, 0, 0, 1, 0)

        # Set lighting material
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (0.4, 0.4, 0.4, 1.0))
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50.0)

        # Draw textured mesh
        glBegin(GL_TRIANGLES)
        for face_idx, face in enumerate(mesh.faces):
            for i in range(3):
                vertex_idx = face[i]
                vertex = mesh.vertices[vertex_idx]
                
                if mesh.face_normals is not None:
                    normal = mesh.face_normals[face_idx]
                    glNormal3fv(normal.astype(np.float32))
                
                if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                    uv = mesh.visual.uv[vertex_idx]
                    glTexCoord2fv(uv.astype(np.float32))

                glVertex3fv(vertex.astype(np.float32))
        glEnd()

        # Save rendered image
        pygame.display.flip()
        pixels = glReadPixels(0, 0, self.window_size[0], self.window_size[1], GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(self.window_size[1], self.window_size[0], 3)
        image = np.flipud(image)

        img = Image.fromarray(image)
        base_name = os.path.splitext(os.path.basename(obj_path))[0]
        output_path = f"renders/{base_name}_view_{random.randint(0, 999999):06d}.png"
        os.makedirs("renders", exist_ok=True)
        img.save(output_path)

        # Clean up texture
        if texture_id:
            glDeleteTextures([texture_id])
            glDisable(GL_TEXTURE_2D)

        return output_path


    def render_all_objects(self):
        """Render all objects in the data directory"""
        rendered_files = []
        
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.obj'):
                    obj_path = os.path.join(root, file)
                    try:
                        output_path = self.render_object(obj_path)
                        rendered_files.append(output_path)
                        print(f"Rendered: {obj_path} -> {output_path}")
                    except Exception as e:
                        print(f"Error rendering {obj_path}: {str(e)}")
        
        return rendered_files

if __name__ == "__main__":
    renderer = ObjectRenderer()
    renderer.setup_camera()
    rendered_files = renderer.render_all_objects()
    print(f"Successfully rendered {len(rendered_files)} objects")