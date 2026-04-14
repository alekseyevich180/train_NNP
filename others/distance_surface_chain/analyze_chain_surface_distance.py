import csv
import math
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_GLOB = "*-ketone.cif"
CHAIN_OUTPUT = SCRIPT_DIR / "chain_surface_distances.csv"
ALPHA_OUTPUT = SCRIPT_DIR / "carbonyl_side_average_distances.csv"


@dataclass
class Atom:
    symbol: str
    label: str
    frac: tuple[float, float, float]
    cart: tuple[float, float, float]


def parse_cif(path: Path) -> tuple[tuple[float, float, float], list[Atom]]:
    lines = path.read_text(encoding="utf-8").splitlines()

    cell_a = cell_b = cell_c = None
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        key, value = parts[0], parts[1]
        if key == "_cell_length_a":
            cell_a = float(value)
        elif key == "_cell_length_b":
            cell_b = float(value)
        elif key == "_cell_length_c":
            cell_c = float(value)

    if cell_a is None or cell_b is None or cell_c is None:
        raise ValueError(f"Failed to read cell lengths from {path}")

    atom_header_start = None
    atom_data_start = None
    for index, line in enumerate(lines):
        if line.strip() == "loop_":
            header_index = index + 1
            headers = []
            while header_index < len(lines) and lines[header_index].strip().startswith("_"):
                headers.append(lines[header_index].strip())
                header_index += 1
            if "_atom_site_type_symbol" in headers and "_atom_site_fract_x" in headers:
                atom_header_start = index
                atom_data_start = header_index
                atom_headers = headers
                break

    if atom_header_start is None or atom_data_start is None:
        raise ValueError(f"Failed to find atom block in {path}")

    symbol_idx = atom_headers.index("_atom_site_type_symbol")
    label_idx = atom_headers.index("_atom_site_label")
    fx_idx = atom_headers.index("_atom_site_fract_x")
    fy_idx = atom_headers.index("_atom_site_fract_y")
    fz_idx = atom_headers.index("_atom_site_fract_z")

    atoms = []
    cell = (cell_a, cell_b, cell_c)
    for line in lines[atom_data_start:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "loop_" or stripped.startswith("_"):
            break
        parts = stripped.split()
        if len(parts) <= max(symbol_idx, label_idx, fx_idx, fy_idx, fz_idx):
            continue

        frac = (
            float(parts[fx_idx]),
            float(parts[fy_idx]),
            float(parts[fz_idx]),
        )
        cart = frac_to_cart(frac, cell)
        atoms.append(
            Atom(
                symbol=parts[symbol_idx],
                label=parts[label_idx],
                frac=frac,
                cart=cart,
            )
        )

    return cell, atoms


def frac_to_cart(frac: tuple[float, float, float], cell: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(frac[i] * cell[i] for i in range(3))


def minimum_image_vector(
    frac_a: tuple[float, float, float],
    frac_b: tuple[float, float, float],
    cell: tuple[float, float, float],
) -> tuple[float, float, float]:
    delta = []
    for i in range(3):
        df = frac_a[i] - frac_b[i]
        df -= round(df)
        delta.append(df * cell[i])
    return tuple(delta)


def pbc_distance(atom_a: Atom, atom_b: Atom, cell: tuple[float, float, float]) -> float:
    dx, dy, dz = minimum_image_vector(atom_a.frac, atom_b.frac, cell)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def group_surface_atoms(atoms: list[Atom]) -> list[int]:
    zn_o_indices = [i for i, atom in enumerate(atoms) if atom.symbol in {"Zn", "O"}]
    z_values = sorted((atoms[i].cart[2], i) for i in zn_o_indices)
    if len(z_values) < 2:
        return zn_o_indices

    max_gap = -1.0
    split_index = len(z_values)
    for i in range(len(z_values) - 1):
        gap = z_values[i + 1][0] - z_values[i][0]
        if gap > max_gap:
            max_gap = gap
            split_index = i + 1

    return [index for _, index in z_values[:split_index]]


def build_carbon_graph(carbon_indices: list[int], atoms: list[Atom], cell: tuple[float, float, float]) -> dict[int, list[int]]:
    graph = {index: [] for index in carbon_indices}
    cutoff = 1.85
    for i, left in enumerate(carbon_indices):
        for right in carbon_indices[i + 1:]:
            if pbc_distance(atoms[left], atoms[right], cell) <= cutoff:
                graph[left].append(right)
                graph[right].append(left)
    return graph


def find_carbonyl_pair(
    atoms: list[Atom],
    cell: tuple[float, float, float],
    carbon_indices: list[int],
    surface_indices: list[int],
) -> tuple[int, int]:
    surface_set = set(surface_indices)
    extra_oxygen_indices = [
        i for i, atom in enumerate(atoms)
        if atom.symbol == "O" and i not in surface_set
    ]
    if len(extra_oxygen_indices) != 1:
        raise ValueError(f"Expected one adsorbate oxygen, found {len(extra_oxygen_indices)}")

    oxygen_index = extra_oxygen_indices[0]
    bonded_carbons = [
        carbon_index for carbon_index in carbon_indices
        if pbc_distance(atoms[oxygen_index], atoms[carbon_index], cell) <= 1.35
    ]
    if len(bonded_carbons) != 1:
        raise ValueError(
            f"Expected one carbonyl carbon bonded to {atoms[oxygen_index].label}, "
            f"found {len(bonded_carbons)}"
        )

    return oxygen_index, bonded_carbons[0]


def traverse_chain(graph: dict[int, list[int]], start: int) -> list[int]:
    order = [start]
    previous = None
    current = start
    while True:
        next_nodes = [node for node in graph[current] if node != previous]
        if not next_nodes:
            break
        previous, current = current, next_nodes[0]
        order.append(current)
    return order


def order_chain(graph: dict[int, list[int]], carbonyl_index: int) -> list[int]:
    endpoints = sorted(node for node, neighbors in graph.items() if len(neighbors) == 1)
    if len(endpoints) != 2:
        raise ValueError("Carbon backbone is not a simple linear chain")

    first_order = traverse_chain(graph, endpoints[0])
    second_order = traverse_chain(graph, endpoints[1])
    first_position = first_order.index(carbonyl_index) + 1
    second_position = second_order.index(carbonyl_index) + 1
    return first_order if first_position <= second_position else second_order


def nearest_surface_distance(
    atom_index: int,
    atoms: list[Atom],
    cell: tuple[float, float, float],
    surface_indices: list[int],
) -> float:
    return min(
        pbc_distance(atoms[atom_index], atoms[surface_index], cell)
        for surface_index in surface_indices
    )


def nearest_surface_neighbors(
    atom_index: int,
    atoms: list[Atom],
    cell: tuple[float, float, float],
    surface_indices: list[int],
    top_n: int = 2,
) -> list[tuple[int, float]]:
    ranked = sorted(
        [
            (
            surface_index,
            pbc_distance(atoms[atom_index], atoms[surface_index], cell),
        )
            for surface_index in surface_indices
        ],
        key=lambda item: item[1],
    )
    return ranked[:top_n]


def analyze_structure(path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    cell, atoms = parse_cif(path)
    surface_indices = group_surface_atoms(atoms)
    surface_z_top = max(atoms[i].cart[2] for i in surface_indices)

    carbon_indices = [i for i, atom in enumerate(atoms) if atom.symbol == "C"]
    if len(carbon_indices) != 9:
        raise ValueError(f"Expected 9 carbon atoms in {path.name}, found {len(carbon_indices)}")

    oxygen_index, carbonyl_index = find_carbonyl_pair(atoms, cell, carbon_indices, surface_indices)
    carbon_graph = build_carbon_graph(carbon_indices, atoms, cell)
    chain_order = order_chain(carbon_graph, carbonyl_index)
    carbonyl_position = chain_order.index(carbonyl_index) + 1

    alpha_indices = sorted(carbon_graph[carbonyl_index], key=lambda idx: chain_order.index(idx))
    if len(alpha_indices) != 2:
        raise ValueError(
            f"Expected two carbons adjacent to carbonyl carbon in {path.name}, "
            f"found {len(alpha_indices)}"
        )

    chain_rows = []
    for position, atom_index in enumerate(chain_order, start=1):
        atom = atoms[atom_index]
        height = atom.cart[2] - surface_z_top
        nearest_neighbors = nearest_surface_neighbors(atom_index, atoms, cell, surface_indices, top_n=2)
        first_surface_index, first_distance = nearest_neighbors[0]
        second_surface_index, second_distance = nearest_neighbors[1]
        role = "chain"
        if atom_index == carbonyl_index:
            role = "carbonyl_carbon"
        elif atom_index in alpha_indices:
            role = "alpha_carbon"

        chain_rows.append(
            {
                "structure": path.stem,
                "chain_position": position,
                "atom_label": atom.label,
                "role": role,
                "height_from_top_surface_A": round(height, 6),
                "nearest_surface_atom_label": atoms[first_surface_index].label,
                "nearest_surface_atom_symbol": atoms[first_surface_index].symbol,
                "nearest_surface_atom_distance_A": round(first_distance, 6),
                "second_nearest_surface_atom_label": atoms[second_surface_index].label,
                "second_nearest_surface_atom_symbol": atoms[second_surface_index].symbol,
                "second_nearest_surface_atom_distance_A": round(second_distance, 6),
            }
        )

    alpha_height_mean = sum(atoms[i].cart[2] - surface_z_top for i in alpha_indices) / 2.0
    alpha_nearest_mean = sum(
        nearest_surface_distance(i, atoms, cell, surface_indices) for i in alpha_indices
    ) / 2.0

    summary_row = {
        "structure": path.stem,
        "surface_top_z_A": round(surface_z_top, 6),
        "carbonyl_oxygen_label": atoms[oxygen_index].label,
        "carbonyl_carbon_label": atoms[carbonyl_index].label,
        "carbonyl_position_in_chain": carbonyl_position,
        "alpha_carbon_1_label": atoms[alpha_indices[0]].label,
        "alpha_carbon_2_label": atoms[alpha_indices[1]].label,
        "alpha_carbon_mean_height_A": round(alpha_height_mean, 6),
        "alpha_carbon_mean_nearest_surface_distance_A": round(alpha_nearest_mean, 6),
    }
    return chain_rows, summary_row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cif_files = sorted(SCRIPT_DIR.glob(INPUT_GLOB))
    if not cif_files:
        raise FileNotFoundError(f"No CIF files matched {INPUT_GLOB} in {SCRIPT_DIR}")

    all_chain_rows = []
    all_summary_rows = []
    for cif_file in cif_files:
        chain_rows, summary_row = analyze_structure(cif_file)
        all_chain_rows.extend(chain_rows)
        all_summary_rows.append(summary_row)

    write_csv(CHAIN_OUTPUT, all_chain_rows)
    write_csv(ALPHA_OUTPUT, all_summary_rows)

    print(f"Wrote {CHAIN_OUTPUT}")
    print(f"Wrote {ALPHA_OUTPUT}")
    for row in all_summary_rows:
        print(
            f"{row['structure']}: carbonyl at chain position {row['carbonyl_position_in_chain']}, "
            f"alpha mean height = {row['alpha_carbon_mean_height_A']:.3f} A, "
            f"alpha mean nearest surface distance = "
            f"{row['alpha_carbon_mean_nearest_surface_distance_A']:.3f} A"
        )


if __name__ == "__main__":
    main()
