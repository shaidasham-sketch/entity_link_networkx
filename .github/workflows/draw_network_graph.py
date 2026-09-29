import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DEVICES_FILE = BASE_DIR / "devices.csv"
OPS_FILE = BASE_DIR / "ops_logs.csv"
LINKS_FILE = BASE_DIR / "inferred_device_links.csv"

OUTPUT_FILE = BASE_DIR / "network_topology_hierarchy.png"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading data...")

devices = pd.read_csv(DEVICES_FILE)
ops_logs = pd.read_csv(OPS_FILE)
edges = pd.read_csv(LINKS_FILE)

print(f"Devices loaded: {len(devices)}")
print(f"Ops log nodes loaded: {len(ops_logs)}")
print(f"Inferred relationships loaded: {len(edges)}")


# ============================================================
# 3. NORMALISE COLUMN NAMES
# ============================================================

devices.columns = (
    devices.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

ops_logs.columns = (
    ops_logs.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

edges.columns = (
    edges.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

required_device_columns = [
    "device_id",
    "hostname",
    "device_type",
    "location"
]

required_edge_columns = [
    "device_id",
    "node_id"
]

for column in required_device_columns:

    if column not in devices.columns:
        raise ValueError(
            f"Missing column '{column}' in devices.csv\n"
            f"Available columns: {list(devices.columns)}"
        )


for column in required_edge_columns:

    if column not in edges.columns:
        raise ValueError(
            f"Missing column '{column}' in inferred_device_links.csv\n"
            f"Available columns: {list(edges.columns)}"
        )


# ============================================================
# 5. CLEAN IDS
# ============================================================

devices["device_id"] = devices["device_id"].astype(str).str.strip()

ops_logs["node_id"] = ops_logs["node_id"].astype(str).str.strip()

edges["device_id"] = edges["device_id"].astype(str).str.strip()

edges["node_id"] = edges["node_id"].astype(str).str.strip()


# ============================================================
# 6. CREATE NETWORK LAYER
# ============================================================

def get_network_layer(device_type):

    device_type = str(device_type).lower().strip()

    if device_type == "router":
        return "Core"

    elif device_type == "switch":
        return "Distribution"

    elif device_type == "firewall":
        return "Security"

    elif device_type == "load balancer":
        return "Services"

    else:
        return "Other"


devices["network_layer"] = devices["device_type"].apply(
    get_network_layer
)


# ============================================================
# 7. CREATE GRAPH
# ============================================================

G = nx.DiGraph()


# ============================================================
# 8. ADD LOCATION NODES
# ============================================================

locations = (
    devices["location"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)


for location in locations:

    location_id = f"LOCATION::{location}"

    G.add_node(
        location_id,
        node_type="location",
        label=location
    )


# ============================================================
# 9. ADD DEVICE NODES
# ============================================================

for _, device in devices.iterrows():

    device_id = str(device["device_id"])

    hostname = str(
        device.get("hostname", device_id)
    )

    device_type = str(
        device.get("device_type", "Unknown")
    )

    location = str(
        device.get("location", "Unknown")
    )

    network_layer = str(
        device.get("network_layer", "Other")
    )

    G.add_node(
        device_id,
        node_type="device",
        label=f"{device_id}\n{hostname}",
        hostname=hostname,
        device_type=device_type,
        location=location,
        network_layer=network_layer
    )


# ============================================================
# 10. ADD OPS LOG NODES
# ============================================================

for _, log in ops_logs.iterrows():

    node_id = str(log["node_id"])

    hostname = str(
        log.get("hostname", node_id)
    )

    event_type = str(
        log.get("event_type", "")
    )

    G.add_node(
        node_id,
        node_type="ops",
        label=f"{node_id}\n{hostname}",
        event_type=event_type
    )


# ============================================================
# 11. ADD HIERARCHY EDGES
# ============================================================

for _, device in devices.iterrows():

    device_id = str(device["device_id"])

    location = str(
        device.get("location", "Unknown")
    )

    location_id = f"LOCATION::{location}"

    # Location → Device
    G.add_edge(
        location_id,
        device_id,
        relationship="CONTAINS"
    )


# ============================================================
# 12. ADD INFERRED AI RELATIONSHIPS
# ============================================================

for _, link in edges.iterrows():

    device_id = str(link["device_id"])
    node_id = str(link["node_id"])

    # Only add if both nodes exist
    if device_id not in G.nodes:
        continue

    if node_id not in G.nodes:
        continue

    score = link.get(
        "similarity_score",
        None
    )

    confidence = link.get(
        "confidence",
        "UNKNOWN"
    )

    try:
        score = float(score)
    except:
        score = 0.0

    G.add_edge(
        device_id,
        node_id,
        relationship="AI_MATCH",
        similarity=score,
        confidence=str(confidence)
    )


# ============================================================
# 13. PRINT GRAPH INFORMATION
# ============================================================

print()
print("Graph created")
print("------------------------------")
print(f"Total nodes: {G.number_of_nodes()}")
print(f"Total edges: {G.number_of_edges()}")
print()


# ============================================================
# 14. CREATE HIERARCHICAL POSITIONS
# ============================================================

pos = {}

location_spacing = 12
device_spacing = 3

for location_index, location in enumerate(locations):

    location_id = f"LOCATION::{location}"

    base_x = location_index * location_spacing

    # --------------------------------------------------------
    # LOCATION POSITION
    # --------------------------------------------------------

    pos[location_id] = (
        base_x,
        0
    )

    # --------------------------------------------------------
    # DEVICES AT THIS LOCATION
    # --------------------------------------------------------

    location_devices = devices[
        devices["location"].astype(str).str.strip()
        == location
    ]

    for device_index, (_, device) in enumerate(
        location_devices.iterrows()
    ):

        device_id = str(
            device["device_id"]
        )

        device_y = -3

        device_x = (
            base_x
            + (
                device_index
                - (len(location_devices) - 1) / 2
            )
            * device_spacing
        )

        pos[device_id] = (
            device_x,
            device_y
        )

        # ----------------------------------------------------
        # OPS NODES LINKED TO DEVICE
        # ----------------------------------------------------

        linked_nodes = edges[
            edges["device_id"] == device_id
        ]

        for node_index, (_, link) in enumerate(
            linked_nodes.iterrows()
        ):

            node_id = str(
                link["node_id"]
            )

            node_x = (
                device_x
                + (
                    node_index
                    - (len(linked_nodes) - 1) / 2
                ) * 1.5
            )

            node_y = -6

            if node_id in G.nodes:

                pos[node_id] = (
                    node_x,
                    node_y
                )


# ============================================================
# 15. FALLBACK POSITIONS
# ============================================================

# Make sure every graph node has a position.

missing_nodes = [
    node
    for node in G.nodes
    if node not in pos
]

if missing_nodes:

    print(
        f"Warning: {len(missing_nodes)} nodes "
        "did not have a hierarchy position."
    )

    fallback_pos = nx.spring_layout(
        G,
        seed=42
    )

    for node in missing_nodes:

        pos[node] = fallback_pos[node]


# ============================================================
# 16. CREATE FIGURE
# ============================================================

plt.figure(
    figsize=(22, 12)
)


# ============================================================
# 17. SEPARATE NODE TYPES
# ============================================================

location_nodes = [
    node
    for node, data in G.nodes(data=True)
    if data.get("node_type") == "location"
]

device_nodes = [
    node
    for node, data in G.nodes(data=True)
    if data.get("node_type") == "device"
]

ops_nodes = [
    node
    for node, data in G.nodes(data=True)
    if data.get("node_type") == "ops"
]


# ============================================================
# 18. DRAW LOCATION NODES
# ============================================================

nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=location_nodes,
    node_shape="o",
    node_size=1000,
    node_color="lightgray",
    edgecolors="black",
    linewidths=2
)


# ============================================================
# 19. DRAW DEVICE NODES
# ============================================================

nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=device_nodes,
    node_shape="s",
    node_size=700,
    node_color="skyblue",
    edgecolors="black",
    linewidths=1.5
)


# ============================================================
# 20. DRAW OPS NODES
# ============================================================

nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=ops_nodes,
    node_shape="D",
    node_size=500,
    node_color="lightgreen",
    edgecolors="black",
    linewidths=1.5
)


# ============================================================
# 21. SEPARATE EDGE TYPES
# ============================================================

hierarchy_edges = []

ai_edges = []

for u, v, data in G.edges(data=True):

    relationship = data.get(
        "relationship",
        ""
    )

    if relationship == "CONTAINS":

        hierarchy_edges.append(
            (u, v)
        )

    elif relationship == "AI_MATCH":

        ai_edges.append(
            (u, v)
        )


# ============================================================
# 22. DRAW HIERARCHY EDGES
# ============================================================

nx.draw_networkx_edges(
    G,
    pos,
    edgelist=hierarchy_edges,
    arrows=True,
    arrowsize=15,
    width=2
)


# ============================================================
# 23. DRAW AI MATCH EDGES
# ============================================================

nx.draw_networkx_edges(
    G,
    pos,
    edgelist=ai_edges,
    arrows=True,
    arrowsize=20,
    width=3,
    style="dashed"
)


# ============================================================
# 24. NODE LABELS
# ============================================================

labels = {}

for node, data in G.nodes(data=True):

    labels[node] = data.get(
        "label",
        node
    )


nx.draw_networkx_labels(
    G,
    pos,
    labels=labels,
    font_size=8,
    font_weight="bold"
)


# ============================================================
# 25. AI MATCH LABELS
# ============================================================

edge_labels = {}

for u, v, data in G.edges(data=True):

    if data.get("relationship") == "AI_MATCH":

        score = data.get(
            "similarity",
            0
        )

        confidence = data.get(
            "confidence",
            ""
        )

        edge_labels[
            (u, v)
        ] = (
            f"{score:.2f}\n"
            f"{confidence}"
        )


nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=edge_labels,
    font_size=7
)


# ============================================================
# 26. TITLE
# ============================================================

plt.title(
    "AI-Assisted Network Topology\n"
    "Location → Device → Ops Node",
    fontsize=18,
    fontweight="bold"
)


# ============================================================
# 27. LEGEND
# ============================================================

plt.text(
    0.01,
    0.02,
    "○ Location    "
    "□ Device    "
    "◇ Ops Node    "
    "— Hierarchy    "
    "- - AI inferred relationship",
    transform=plt.gca().transAxes,
    fontsize=10
)


# ============================================================
# 28. CLEAN AXES
# ============================================================

plt.axis("off")

plt.tight_layout()


# ============================================================
# 29. SAVE GRAPH
# ============================================================

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

print()
print(
    f"Graph saved to:\n{OUTPUT_FILE}"
)


# ============================================================
# 30. DISPLAY
# ============================================================

plt.show()
