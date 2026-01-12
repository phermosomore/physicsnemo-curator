# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Optional, Dict

import numpy as np
from lasso.dyna import ArrayType, D3plot


def compute_node_type(pos_raw: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """
    Identify structural vs wall nodes based on displacement variation.

    Args:
        pos_raw: (timesteps, num_nodes, 3) raw displacement trajectories
        threshold: max displacement below which a node is considered "wall"

    Returns:
        node_type: (num_nodes,) uint8 array where 1=wall, 0=structure
    """
    variation = np.max(np.abs(pos_raw - pos_raw[0:1, :, :]), axis=0)
    variation = np.max(variation, axis=1)
    is_wall = variation < threshold
    return np.where(is_wall, 1, 0).astype(np.uint8)


def build_edges_from_mesh_connectivity(mesh_connectivity) -> set:
    """
    Build unique edges from mesh connectivity.

    Args:
        mesh_connectivity: list of elements (list[int])

    Returns:
        Set of unique edges (i,j)
    """
    edges = set()
    for cell in mesh_connectivity:
        n = len(cell)
        for idx in range(n):
            edges.add(tuple(sorted((cell[idx], cell[(idx + 1) % n]))))
    return edges


def load_d3plot_data(data_path: str):
    """Load node coordinates and displacements from a d3plot file."""
    # Note: OpenRadioss simulations often output to D3PLOT format. 
    # This function assumes the result file is a valid d3plot.
    dp = D3plot(data_path)
    coords = dp.arrays[ArrayType.node_coordinates]  # (num_nodes, 3)
    pos_raw = dp.arrays[ArrayType.node_displacement]  # (timesteps, num_nodes, 3)
    mesh_connectivity = dp.arrays[ArrayType.element_shell_node_indexes]
    part_ids = dp.arrays[ArrayType.element_shell_part_indexes]

    # Get actual part IDs if available
    actual_part_ids = None
    if ArrayType.part_ids in dp.arrays:
        actual_part_ids = dp.arrays[ArrayType.part_ids]

    return coords, pos_raw, mesh_connectivity, part_ids, actual_part_ids


def find_k_file(run_dir: Path) -> Optional[Path]:
    """Find .k file in run directory."""
    k_files = list(run_dir.glob("*.k"))
    return k_files[0] if k_files else None


def find_rad_file(run_dir: Path) -> Optional[Path]:
    """Find OpenRadioss starter file (*0000.rad) in run directory.
    
    Args:
        run_dir: Path to run directory.

    Returns:
        Path to .rad file or None.
    """
    # OpenRadioss starter files usually end in 0000.rad
    rad_files = list(run_dir.glob("*0000.rad"))
    return rad_files[0] if rad_files else None

def parse_k_file(k_file_path: Path) -> dict[int, float]:
    """Parse LS-DYNA .k file to extract part thickness information.

    Args:
        k_file_path: Path to .k file.

    Returns:
        Dictionary mapping part ID to thickness.
    """

    part_to_section = {}
    section_thickness = {}

    with open(k_file_path, "r") as f:
        lines = [
            line.strip() for line in f if line.strip() and not line.startswith("$")
        ]

    i = 0
    while i < len(lines):
        line = lines[i]
        if "*PART" in line.upper():
            # After *PART:
            # i+1 = part name (skip)
            # i+2 = part id, section id, material id
            if i + 2 < len(lines):
                tokens = lines[i + 2].split()
                if len(tokens) >= 2:
                    part_id = int(tokens[0])
                    section_id = int(tokens[1])
                    part_to_section[part_id] = section_id
            i += 3
        elif "*SECTION_SHELL" in line.upper():
            # Multiple sections can be defined under one *SECTION_SHELL keyword
            # Each section has two lines: header line and thickness line
            i += 1  # Skip the *SECTION_SHELL line
            while i < len(lines) and not lines[i].startswith("*"):
                # Check if this line looks like a section header (starts with a number)
                if i < len(lines) and lines[i].strip() and lines[i][0].isdigit():
                    header_line = lines[i]
                    thickness_line = lines[i + 1] if i + 1 < len(lines) else ""

                    # Extract section ID from header line (first number)
                    header_tokens = header_line.split()
                    if len(header_tokens) >= 1:
                        try:  # noqa: PERF203
                            section_id = int(header_tokens[0])
                        except ValueError:
                            section_id = None
                    else:
                        section_id = None

                    # Extract thickness values from thickness line
                    thickness_values = []
                    thickness_tokens = thickness_line.split()
                    for t in thickness_tokens:
                        try:
                            thickness_values.append(float(t))
                        except ValueError:  # noqa: PERF203
                            thickness_values.append(0.0)
                    # Calculate average thickness (ignore zeros)
                    non_zero_thicknesses = [t for t in thickness_values if t > 0.0]
                    if non_zero_thicknesses:
                        thickness = sum(non_zero_thicknesses) / len(
                            non_zero_thicknesses
                        )
                    elif thickness_values:
                        thickness = sum(thickness_values) / len(thickness_values)
                    else:
                        thickness = 0.0
                    if section_id is not None:
                        section_thickness[section_id] = thickness

                    i += 2  # Skip both header and thickness lines
                else:
                    i += 1
        else:
            i += 1

    part_thickness = {
        pid: section_thickness.get(sid, 0.0) for pid, sid in part_to_section.items()
    }
    return part_thickness

def parse_rad_file(rad_file_path: Path) -> dict[int, float]:
    """Parse OpenRadioss .rad file to extract part thickness information.

    Links /PART/ IDs to /PROP/SHELL/ IDs to extract thickness.
    """
    part_to_prop = {}
    prop_thickness = {}

    with open(rad_file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f]

    i = 0
    while i < len(lines):
        line = lines[i]
        
        # --- Parse Parts ---
        if line.startswith("/PART/"):
            try:
                parts = line.split("/")
                if len(parts) >= 3 and parts[2].isdigit():
                    part_id = int(parts[2])
                    
                    # Usually:
                    # i   : /PART/13
                    # i+1 : Title (Skip)
                    # i+2 : Data (PropID is 1st token)
                    
                    data_line_idx = i + 2
                    if data_line_idx < len(lines):
                        tokens = lines[data_line_idx].split()
                        # Verify this looks like data (starts with digit)
                        if tokens and tokens[0].isdigit():
                            prop_id = int(tokens[0])
                            part_to_prop[part_id] = prop_id
            except (ValueError, IndexError):
                pass
            i += 1

        # --- Parse Shell Properties ---
        elif line.startswith("/PROP/SHELL/"):
            try:
                parts = line.split("/")
                if len(parts) >= 4 and parts[3].isdigit():
                    prop_id = int(parts[3])
                    
                    # Structure:
                    # i   : /PROP/SHELL/1
                    # i+1 : Title (e.g., "Prop_DP600") -> SKIP THIS
                    # i+2 : Data Line 1 (Ishell...)
                    # i+3 : Data Line 2 (hm...)
                    # i+4 : Data Line 3 (Thick is at index 2)
                    
                    current_idx = i + 2 # Start looking AFTER the title
                    data_lines_found = 0
                    
                    while current_idx < len(lines) and data_lines_found < 3:
                        curr_line = lines[current_idx]
                        
                        # Stop if we hit a new keyword
                        if curr_line.startswith("/"):
                            break
                            
                        # Skip comments and empty lines
                        if not curr_line or curr_line.startswith("#"):
                            current_idx += 1
                            continue
                        
                        data_lines_found += 1
                        
                        # Thickness is on the 3rd valid data line
                        if data_lines_found == 3:
                            tokens = curr_line.split()
                            # Format: N Istrain Thick ...
                            # Thick is index 2
                            if len(tokens) >= 3:
                                try:
                                    thick_val = float(tokens[2])
                                    prop_thickness[prop_id] = thick_val
                                except ValueError:
                                    pass
                        current_idx += 1
            except (ValueError, IndexError):
                pass
            i += 1
        else:
            i += 1

    # Map Parts to Thickness
    part_thickness = {}
    for pid, prid in part_to_prop.items():
        if prid in prop_thickness:
            part_thickness[pid] = prop_thickness[prid]
        else:
            # If property not found, default to 0.0 or log warning if needed
            part_thickness[pid] = 0.0
            
    return part_thickness


def compute_node_thickness(
    mesh_connectivity: np.ndarray,
    part_ids: np.ndarray,
    part_thickness_map: dict[int, float],
    actual_part_ids: np.ndarray = None,
    num_nodes: int = None,
) -> np.ndarray:
    """
    Compute thickness for each node. Handles 1-based indexing automatically.

    Args:
        mesh_connectivity: (num_elements, n_nodes) Element connectivity
        part_ids: (num_elements,) Part IDs for each element
        part_thickness_map: Mapping from part ID to thickness
        actual_part_ids: Actual part IDs if available
        num_nodes: Total number of nodes (len(coords)). If provided, ensures output matches.
    """
    # 1. Resolve Part IDs
    if actual_part_ids is not None:
        # Skip index 0 in actual_part_ids as it's often a placeholder in d3plot arrays
        part_index_to_id = {
            i: pid for i, pid in enumerate(actual_part_ids) if i > 0
        }
    else:
        # Fallback if no actual mapping exists
        sorted_part_ids = sorted(part_thickness_map.keys())
        part_index_to_id = {i: pid for i, pid in enumerate(sorted_part_ids, 1)}

    # 2. Map Element Thickness
    element_thickness = np.zeros(len(part_ids))
    for i, part_index in enumerate(part_ids):
        actual_part_id = part_index_to_id.get(part_index)
        if actual_part_id is not None:
            element_thickness[i] = part_thickness_map.get(actual_part_id, 0.0)

    # 3. Handle Connectivity Indexing (1-based vs 0-based)
    # If we weren't given a node count, guess it from the max index
    if num_nodes is None:
        max_idx = 0
        for element in mesh_connectivity:
            max_idx = max(max_idx, np.max(element))
        num_nodes = max_idx + 1

    # Initialize arrays
    node_thickness = np.zeros(num_nodes)
    node_thickness_count = np.zeros(num_nodes)

    # Check for 1-based indexing
    # If the max index matches the size (index == size), we definitely have an offset
    # or if min index is 1, it's likely 1-based.
    conn_min = np.min(mesh_connectivity)
    conn_max = np.max(mesh_connectivity)
    
    offset = 0
    if conn_max >= num_nodes:
        if conn_min >= 1:
            # Shift down by 1
            offset = -1 
        else:
            # Hybrid/Error state: indices are out of bounds but start at 0. 
            # We must clip to avoid crashing.
            pass 

    # 4. Accumulate Thickness
    # We iterate carefully to avoid crashing on bad indices
    for i, element in enumerate(mesh_connectivity):
        thick = element_thickness[i]
        for node_idx in element:
            # Apply offset (fix 1-based)
            idx = int(node_idx + offset)
            
            # Boundary check - ignore nodes that don't exist in coords
            if 0 <= idx < num_nodes:
                node_thickness[idx] += thick
                node_thickness_count[idx] += 1

    # 5. Average
    mask = node_thickness_count > 0
    node_thickness[mask] /= node_thickness_count[mask]

    return node_thickness