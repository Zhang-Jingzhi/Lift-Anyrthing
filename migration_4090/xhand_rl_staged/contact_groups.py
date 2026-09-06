"""Map the canonical 60 object-contact filters into twelve semantic groups."""

from __future__ import annotations

from migration_4090.xhand_rl.contact_layout import contact_filter_layout


CONTACT_GROUPS = ("palm", "index", "middle", "ring", "pinky", "thumb")


def body_contact_group(body_name: str) -> str:
    if "_index_" in body_name:
        return "index"
    if "_mid_" in body_name:
        return "middle"
    if "_ring_" in body_name:
        return "ring"
    if "_pinky_" in body_name:
        return "pinky"
    if "_thumb_" in body_name:
        return "thumb"
    return "palm"


def grouped_filter_indices() -> dict[tuple[str, str], list[int]]:
    groups = {(side, group): [] for side in ("left", "right") for group in CONTACT_GROUPS}
    for index, (side, body_name, _) in enumerate(contact_filter_layout("/World/Robot")):
        groups[(side, body_contact_group(body_name))].append(index)
    if any(not indices for indices in groups.values()):
        missing = [key for key, indices in groups.items() if not indices]
        raise RuntimeError(f"empty contact groups: {missing}")
    return groups
