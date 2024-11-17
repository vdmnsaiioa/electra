import ase
import numpy as np
from scipy.spatial.transform import Rotation
from itertools import combinations, permutations
import plotly.graph_objects as go
import torch

def is_close(a, b, rtol=0.2, atol=0.025):
    """Check if two values or arrays are close within tolerance."""
    return np.allclose(a, b, rtol=rtol, atol=atol)

def normalize(v):
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def find_reflection_plane(points, center, normal):
    """
    Check if a plane through center with given normal is a reflection plane.
    Returns True if the plane is a reflection plane.
    """
    normal = normalize(normal)
    reflected_points = points.copy()
    
    for i in range(len(points)):
        # Vector from center to point
        v = points[i] - center
        # Reflection formula: r = v - 2(v·n)n
        reflected_points[i] = points[i] - 2 * np.dot(v, normal) * normal
    
    # Sort points by distance from center to match them efficiently
    orig_dists = np.linalg.norm(points - center, axis=1)
    refl_dists = np.linalg.norm(reflected_points - center, axis=1)
    
    # If the sets of distances don't match, this can't be a reflection plane
    if not is_close(np.sort(orig_dists), np.sort(refl_dists)):
        return False
    
    # For each reflected point, find a matching original point
    used = set()
    for i in range(len(points)):
        found_match = False
        for j in range(len(points)):
            if j not in used and is_close(reflected_points[i], points[j]):
                used.add(j)
                found_match = True
                break
        if not found_match:
            return False
    return True

def find_inversion_center(points, center):
    """
    Check if center is an inversion center.
    Returns True if it is an inversion center.
    """
    inverted_points = 2 * center - points
    
    # Sort points by distance from center to match them efficiently
    orig_dists = np.linalg.norm(points - center, axis=1)
    inv_dists = np.linalg.norm(inverted_points - center, axis=1)
    
    # If the sets of distances don't match, this can't be an inversion center
    if not is_close(np.sort(orig_dists), np.sort(inv_dists)):
        return False
    
    # For each inverted point, find a matching original point
    used = set()
    for i in range(len(points)):
        found_match = False
        for j in range(len(points)):
            if j not in used and is_close(inverted_points[i], points[j]):
                used.add(j)
                found_match = True
                break
        if not found_match:
            return False
    return True

def find_all_reflection_planes(points, center):
    """
    Find all reflection planes through the center.
    Returns list of normal vectors to reflection planes.
    """
    reflection_planes = []
    
    # Try planes defined by pairs of points
    for i, j in combinations(range(len(points)), 2):
        v = points[j] - points[i]
        if np.any(v):  # Skip if points are identical
            normal = normalize(v)
            if find_reflection_plane(points, center, normal):
                reflection_planes.append(normal)
    
    # Try planes perpendicular to lines between points
    for i, j in combinations(range(len(points)), 2):
        v = points[j] - points[i]
        if np.any(v):
            # Create vectors perpendicular to v
            if abs(v[2]) > 1e-10:
                u = np.array([1, 1, -(v[0] + v[1])/v[2]])
            elif abs(v[1]) > 1e-10:
                u = np.array([1, -(v[0] + v[2])/v[1], 1])
            else:
                u = np.array([-(v[1] + v[2])/v[0], 1, 1])
            
            u = normalize(u)
            if find_reflection_plane(points, center, u):
                reflection_planes.append(u)

    normals = []
    n_points = len(points)

    # Generate normals from pairs of points
    for i, j in combinations(range(n_points), 2):
        v = points[j] - points[i]
        if np.linalg.norm(v) > 1e-6:  # Avoid degenerate cases
            normal = normalize(v)
            normals.append(normal)

    # Ensure unique normals (accounting for ±normal equivalence)
    unique_normals = []
    for normal in normals:
        if not any(is_close(normal, n) or is_close(normal, -n) for n in unique_normals):
            unique_normals.append(normal)

    for normal in unique_normals:
        if find_reflection_plane(points, center, normal):
            reflection_planes.append(normal)

    
    # Remove duplicates (accounting for ±normal being equivalent)
    unique_planes = []
    for plane in reflection_planes:
        if not any(is_close(plane, p) or is_close(plane, -p) 
                  for p in unique_planes):
            unique_planes.append(plane)
    
    return unique_planes

def analyze_point_group_symmetry(points, center):
    """
    Analyze point group symmetry operations through a specified center point.
    
    Args:
        points: numpy array of shape (n, 3) containing point coordinates
        center: numpy array of shape (3,) specifying the center point
    
    Returns:
        dict containing:
        - reflection_planes: list of normal vectors to reflection planes
        - has_inversion: boolean indicating if center is an inversion center
        - reflection_decomposition: if has_inversion, gives reflection planes
          whose product gives the inversion
    """
    # Convert inputs to numpy arrays
    points = np.array(points)
    center = np.array(center)
    
    # Find all reflection planes
    reflection_planes = find_all_reflection_planes(points, center)
    
    # Check for inversion center
    has_inversion = find_inversion_center(points, center)
    
    # If there's an inversion, find reflection decomposition
    reflection_decomposition = None
    if has_inversion and len(reflection_planes) >= 2:
        # Inversion can be decomposed as product of two perpendicular reflections
        for i, j in combinations(range(len(reflection_planes)), 2):
            if is_close(abs(np.dot(reflection_planes[i], reflection_planes[j])), 0):
                reflection_decomposition = [reflection_planes[i], reflection_planes[j]]
                break
    
    return {
        'reflection_planes': reflection_planes,
        'has_inversion': has_inversion,
        'reflection_decomposition': reflection_decomposition
    }

def plot_molecule_and_symmetry(points, center, reflection_planes, has_inversion, reflection_decomposition):
    """
    Plot the molecule and its symmetry elements using Plotly.
    
    Args:
        points: numpy array of shape (n, 3) containing point coordinates
        center: numpy array of shape (3,) specifying the center point
        reflection_planes: list of normal vectors to reflection planes
        has_inversion: boolean indicating if center is an inversion center
        reflection_decomposition: if has_inversion, list of reflection planes
          whose product gives the inversion
    """
    # Find the bounding box for the molecule
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    
    # Create the figure
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color='black'
                )
            )
        ],
        layout=go.Layout(
            scene=dict(
                xaxis_range=[x_min - 1, x_max + 1],
                yaxis_range=[y_min - 1, y_max + 1],
                zaxis_range=[z_min - 1, z_max + 1],
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            width=800,
            height=600
        )
    )
    
    # Plot reflection planes
    for plane in reflection_planes:
        # Create plane equation coefficients
        a, b, c = plane
        d = -np.dot(plane, center)
        
        # Generate grid based on which component is non-zero
        if abs(c) > 1e-6:  # z-component is non-zero
            x, y = np.meshgrid(np.linspace(x_min - 1, x_max + 1, 50), 
                              np.linspace(y_min - 1, y_max + 1, 50))
            z = (-a * x - b * y - d) / c
        elif abs(b) > 1e-6:  # y-component is non-zero
            x, z = np.meshgrid(np.linspace(x_min - 1, x_max + 1, 50),
                              np.linspace(z_min - 1, z_max + 1, 50))
            y = (-a * x - c * z - d) / b
        else:  # x-component is non-zero (assuming plane normal is not zero vector)
            y, z = np.meshgrid(np.linspace(y_min - 1, y_max + 1, 50),
                              np.linspace(z_min - 1, z_max + 1, 50))
            x = (-b * y - c * z - d) / a

        # Create surface based on which component we solved for
        if abs(c) > 1e-6:
            fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Inferno', opacity=0.5, showscale=False))
        elif abs(b) > 1e-6:
            fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Inferno', opacity=0.5, showscale=False))
        else:
            fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Inferno', opacity=0.5, showscale=False))
        
        # Plot normal vectors
        fig.add_trace(
            go.Scatter3d(
                x=[center[0], center[0] + 0.5 * a],
                y=[center[1], center[1] + 0.5 * b],
                z=[center[2], center[2] + 0.5 * c],
                mode='lines',
                line=dict(
                    color='orange',
                    width=3
                )
            )
        )
    
    # Plot inversion center
    if has_inversion:
        fig.add_trace(
            go.Scatter3d(
                x=[center[0]],
                y=[center[1]],
                z=[center[2]],
                mode='markers',
                marker=dict(
                    size=8,
                    color='red'
                )
            )
        )
    
    # Plot reflection planes for inversion decomposition
    if reflection_decomposition:
        for plane in reflection_decomposition:
            a, b, c = plane
            d = -np.dot(plane, center)
            x, y = np.meshgrid(np.linspace(x_min - 1, x_max + 1, 50), 
                              np.linspace(y_min - 1, y_max + 1, 50))
            z = (-a * x - b * y - d) / c
            
            fig.add_trace(
                go.Surface(
                    x=x, y=y, z=z,
                    colorscale='Inferno',
                    opacity=0.5,
                    showscale=False
                )
            )
            
            fig.add_trace(
                go.Scatter3d(
                    x=[center[0], center[0] + 0.5 * a],
                    y=[center[1], center[1] + 0.5 * b],
                    z=[center[2], center[2] + 0.5 * c],
                    mode='lines',
                    line=dict(
                        color='blue',
                        width=3
                    )
                )
            )
    
    fig.update_layout(
        title='Molecular Symmetry',
        width=800,
        height=600
    )
    
    fig.show()

def get_symmetries(molecule: ase.Atoms):
    vectors = {}
    points = np.array(molecule.positions)
    symbols = molecule.get_chemical_symbols()
    for i, (symbol, atom) in enumerate(zip(symbols, molecule)):
        center = atom.position
        result = analyze_point_group_symmetry(points, center)
        vectors[f"{atom.symbol}_{i}"] = torch.tensor(result['reflection_planes'],
                                                     dtype=torch.float32,
                                                     device='cuda' if torch.cuda.is_available() else 'cpu')
    return vectors


if __name__ == "__main__":
    atoms = ase.Atoms('H2O', positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    result = get_symmetries(atoms)
