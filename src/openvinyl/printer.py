"""Printer profiles and extrusion math.

All lengths in mm, speeds in mm/s, accelerations in mm/s^2.
"""

from dataclasses import dataclass, field, replace
import math


@dataclass
class PrinterProfile:
    name: str = "Bambu P1S"
    bed_x: float = 256.0
    bed_y: float = 256.0
    max_z: float = 250.0
    bed_margin: float = 1.0          # keep-out from bed edge for groove vertices
    filament_diameter: float = 1.75
    nozzle_diameter: float = 0.4
    extrusion_width_factor: float = 1.2   # w_e = k * nozzle (open parameter, k in [1.1, 1.3])
    layer_height: float = 0.2             # groove bead height
    # Motion limits used for the groove pass. Conservative on purpose:
    # the groove is one long precision perimeter, not a speed print.
    groove_speed_max: float = 30.0
    groove_speed_min: float = 4.0
    groove_accel_max: float = 2000.0      # lateral accel budget for the wiggle
    travel_speed: float = 150.0
    z_hop: float = 1.0
    max_volumetric_flow: float = 12.0     # mm^3/s, PLA on a 0.4 nozzle, conservative
    max_segment_rate: float = 500.0       # segments/s the firmware planner can sustain (open parameter)
    min_segment_length: float = 0.05
    extra: dict = field(default_factory=dict)

    @property
    def extrusion_width(self) -> float:
        return self.extrusion_width_factor * self.nozzle_diameter

    @property
    def filament_area(self) -> float:
        return math.pi * (self.filament_diameter / 2.0) ** 2

    def bead_area(self, width: float = None, height: float = None) -> float:
        """Cross-section of a deposited bead: rectangle with semicircular ends
        (the same model slicers use)."""
        w = self.extrusion_width if width is None else width
        h = self.layer_height if height is None else height
        return (w - h) * h + math.pi * (h / 2.0) ** 2

    def e_per_mm(self, width: float = None, height: float = None) -> float:
        """Filament length per mm of path."""
        return self.bead_area(width, height) / self.filament_area

    def volumetric_flow(self, speed: float, width: float = None, height: float = None) -> float:
        return self.bead_area(width, height) * speed

    @property
    def center(self):
        return (self.bed_x / 2.0, self.bed_y / 2.0)

    def in_bounds(self, x: float, y: float) -> bool:
        m = self.bed_margin
        return (m <= x <= self.bed_x - m) and (m <= y <= self.bed_y - m)


PROFILES = {
    "p1s": PrinterProfile(),
    "p1s-0.2": PrinterProfile(name="Bambu P1S 0.2mm", nozzle_diameter=0.2,
                              layer_height=0.12, max_volumetric_flow=4.0),
}


def load_profile(path: str) -> PrinterProfile:
    """A profile from a JSON file of PrinterProfile fields (see printers/template.json and
    docs/ADAPTING_YOUR_PRINTER.md). Fields left out keep the P1S defaults; an unknown key is an error, so a typo
    can't silently fall back to the P1S number. Keys starting with "_" are notes and are ignored."""
    import json
    from dataclasses import fields
    with open(path) as fh:
        d = json.load(fh)
    d = {k: v for k, v in d.items() if not k.startswith("_")}
    known = {f.name for f in fields(PrinterProfile)}
    bad = sorted(set(d) - known)
    if bad:
        raise KeyError(f"{path}: unknown profile fields {bad}; valid: {sorted(known)}")
    return PrinterProfile(**d)


def get_profile(name: str = "p1s", nozzle: float = None) -> PrinterProfile:
    """A built-in profile by name, or a JSON profile by path (anything ending in .json)."""
    if name.lower().endswith(".json"):
        p = load_profile(name)
        if nozzle is not None:
            p.nozzle_diameter = nozzle
        return p
    key = name.lower()
    if key not in PROFILES:
        raise KeyError(f"unknown printer profile '{name}', choose from {sorted(PROFILES)} or pass a .json file")
    p = replace(PROFILES[key])
    if nozzle is not None:
        p.nozzle_diameter = nozzle
    return p
