# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class CrashMetadata:
    """Metadata for Crash simulation data.

    Version history:
    - 1.0: Initial version with expected metadata fields.
    - 1.1: Added boundary_conditions for simulation parameters (velocity, wall geometry, etc.)
    """

    # Simulation identifiers
    filename: str

    # Boundary conditions extracted from run JSON file
    # Keys are configurable (e.g., velocity_vector, rwall_diameter, rwall_origin)
    boundary_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrashExtractedDataInMemory:
    """Container for Crash simulation data and metadata extracted from the simulation.

    Version history:
    - 1.0: Initial version with expected data fields.
    - 1.1: Added boundary_conditions support via metadata.
    """

    # Metadata
    metadata: CrashMetadata = None

    # Raw data
    pos_raw: np.ndarray = None
    mesh_connectivity: np.ndarray = None
    node_thickness: np.ndarray = None

    # Processed data
    filtered_pos_raw: np.ndarray = None
    filtered_mesh_connectivity: np.ndarray = None
    filtered_node_thickness: np.ndarray = None
    edges: np.ndarray = None
