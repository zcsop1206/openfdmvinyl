"""Command line: info | generate | preview."""

import argparse
import json
import os
import sys

import numpy as np

from . import __version__
from .printer import get_profile
from .groove import GrooveParams, build_groove
from .dsp import read_wav, condition, test_signal, write_wav
from .gcode import emit_groove, generic_start, generic_end
from .splice import analyze_base, splice
from .preview import preview_file


def _add_geometry_args(ap):
    ap.add_argument("--printer", default="p1s",
                    help="built-in profile (p1s, p1s-0.2) or a .json profile file (see printers/template.json)")
    ap.add_argument("--nozzle", type=float, default=None, help="override nozzle diameter, mm")
    ap.add_argument("--rpm", type=float, default=78.0)
    ap.add_argument("--diameter", type=float, default=254.0, help="record diameter, mm")
    ap.add_argument("--outer", type=float, default=None, help="outer groove radius, mm (default diameter/2 - 7)")
    ap.add_argument("--inner", type=float, default=40.0, help="inner groove radius, mm")
    ap.add_argument("--amplitude", type=float, default=0.10, help="peak lateral modulation, mm")
    ap.add_argument("--bead-spacing", type=float, default=1.1, help="bead centre spacing as multiple of w_e")
    ap.add_argument("--land", type=float, default=None, help="gap between turns, mm (default w_e)")
    ap.add_argument("--segment", type=float, default=0.10, help="arc length per G-code segment, mm")
    ap.add_argument("--theta-max", type=float, default=45.0, help="max groove slope, deg")
    ap.add_argument("--layers", type=int, default=1, help="bead layers per groove")
    ap.add_argument("--extra-layers-from", type=float, default=None,
                    help="play time, s, where the layers above the first start (default: the whole groove)")
    ap.add_argument("--floor-layers", type=int, default=0,
                    help="spiral floor layers printed under the groove band, following the groove")
    ap.add_argument("--schedule", default=None,
                    help="JSON file: bands along the groove ([{t1, spacing, layers, floor}, ...], audio time), "
                         "overriding --bead-spacing, --layers, --extra-layers-from and --floor-layers")
    ap.add_argument("--ramp", type=float, default=0.75, help="with --schedule: band change length, s")
    ap.add_argument("--stylus", type=float, default=0.076, help="stylus tip radius, mm")
    ap.add_argument("--lead-in", type=float, default=1.0, help="unmodulated turns at start")
    ap.add_argument("--lead-out", type=float, default=1.0, help="unmodulated turns at end")
    ap.add_argument("--speed-max", type=float, default=None)
    ap.add_argument("--speed-min", type=float, default=None)
    ap.add_argument("--accel", type=float, default=None, help="lateral accel budget, mm/s^2")


def _load_schedule(args):
    path = getattr(args, "schedule", None)
    if not path:
        return None
    with open(path) as fh:
        return json.load(fh)


def _params_from(args):
    printer = get_profile(args.printer, args.nozzle)
    if args.speed_max is not None:
        printer.groove_speed_max = args.speed_max
    if args.speed_min is not None:
        printer.groove_speed_min = args.speed_min
    if args.accel is not None:
        printer.groove_accel_max = args.accel
    outer = args.outer if args.outer is not None else args.diameter / 2.0 - 7.0
    gp = GrooveParams(rpm=args.rpm, record_diameter=args.diameter,
                      outer_groove_radius=outer, inner_groove_radius=args.inner,
                      amplitude=args.amplitude, bead_spacing_factor=args.bead_spacing,
                      land=args.land, lead_in_turns=args.lead_in, lead_out_turns=args.lead_out,
                      segment_length=args.segment, theta_max_deg=args.theta_max,
                      groove_layers=args.layers, stylus_radius=args.stylus,
                      extra_layers_from_s=args.extra_layers_from, floor_layers=args.floor_layers,
                      schedule=_load_schedule(args), ramp_s=getattr(args, "ramp", 0.75))
    return printer, gp


def cmd_info(args):
    printer, gp = _params_from(args)
    g = gp.resolve(printer)
    print(f"OpenVinyl {__version__}  printer={printer.name}  nozzle={printer.nozzle_diameter} mm")
    for k, v in g.summary().items():
        print(f"  {k:32s} {v}")


def cmd_generate(args):
    printer, gp = _params_from(args)
    g = gp.resolve(printer)
    lpf = args.lpf if args.lpf else g.f_smear(gp.inner_groove_radius)

    if args.input:
        audio, fs = read_wav(args.input)
        print(f"input: {args.input}  {len(audio)/fs:.1f} s @ {fs} Hz")
    else:
        fs = 44100
        dur = args.duration if args.duration else g.audio_time_available
        audio = test_signal(args.signal, dur, fs)
        print(f"test signal: {args.signal}  {dur:.1f} s")
    if args.duration and args.input:
        audio = audio[: int(args.duration * fs)]

    if args.signal and args.signal == "silence":
        cond = audio
    elif args.no_dsp:
        p = np.abs(audio).max()
        cond = audio / p if p > 0 else audio
    else:
        cond = condition(audio, fs, hpf=args.hpf, lpf=lpf, ratio=args.ratio,
                         target_dr_db=args.target_dr)
        print(f"dsp: HPF {args.hpf:.0f} Hz, LPF {lpf:.0f} Hz, ratio {args.ratio}:1")

    path = build_groove(cond, fs, gp, printer)
    if "truncated_audio_s" in path.stats:
        print(f"WARNING: audio truncated by {path.stats['truncated_audio_s']} s "
              f"(disc holds {g.audio_time_available:.1f} s at {gp.rpm} RPM)")

    center = (None, None)
    z_top = args.z_top
    base_lines = None
    info = None
    if args.base:
        with open(args.base, "r", errors="replace") as fh:
            base_lines = fh.read().splitlines()
        info = analyze_base(base_lines)
        z_top = info["top_z"] if z_top is None else z_top
        center = info["center"] if info["center"] else center
        print(f"base: top Z {z_top} mm, centre {tuple(round(c, 2) for c in center)}, "
              f"E mode {info['extrusion_mode']}, insert before: {info['end_marker']!r}")
        cfg = info["config"]
        if "nozzle_diameter" in cfg:
            try:
                nd = float(cfg["nozzle_diameter"].split(",")[0])
                if abs(nd - printer.nozzle_diameter) > 1e-6:
                    print(f"WARNING: base sliced for {nd} mm nozzle, groove assumes "
                          f"{printer.nozzle_diameter} mm")
            except ValueError:
                pass
        if info.get("last_layer_extent"):
            ex = info["last_layer_extent"]
            if abs(max(ex) - gp.record_diameter) > 3.0:
                print(f"WARNING: base top layer extent {ex[0]:.1f} x {ex[1]:.1f} mm, "
                      f"record diameter set to {gp.record_diameter} mm")
    if z_top is None:
        z_top = 1.0
        print("no --base and no --z-top: assuming disc top at Z=1.0 mm")
    if args.center:
        center = tuple(float(c) for c in args.center.split(","))

    block, report = emit_groove(path, printer, z_top, center)

    if base_lines is not None:
        out_lines, info = splice(base_lines, block, info)
        if info.get("total_time_s"):
            print(f"print time: base {info['base_time_s'] / 60:.1f} min + groove {info['groove_time_s'] / 60:.1f} min = "
                  f"{info['total_time_s'] / 60:.1f} min; groove filament {info['groove_filament_g']:.1f} g "
                  f"(header, progress lines and .3mf updated)")
    elif args.standalone:
        out_lines = generic_start() + block + generic_end()
    else:
        out_lines = block

    out = args.output or _default_out(args)
    with open(out, "w", newline="\n") as fh:
        fh.write("\n".join(out_lines) + "\n")
    with open(os.path.splitext(out)[0] + ".report.json", "w") as fh:
        json.dump({"geometry": g.summary(), "report": report.to_dict(),
                   "dsp": {"hpf": args.hpf, "lpf": lpf, "ratio": args.ratio}}, fh, indent=2)
    if args.save_conditioned:
        write_wav(args.save_conditioned, cond, fs)

    print(f"wrote {out}  ({len(out_lines):,} lines)")
    if getattr(args, "base_3mf", None):
        if base_lines is None:
            sys.exit("--base-3mf needs --base (the G-code inside that same .3mf)")
        from .splice import pack_3mf
        pack_3mf(args.base_3mf, "\n".join(out_lines) + "\n", out + ".3mf", info)
        print(f"wrote {out}.3mf")
    print("validation:" if report.ok else "validation FAILED:")
    print(report.text())
    if not report.ok:
        sys.exit(2)


def _default_out(args):
    if args.input:
        return os.path.splitext(args.input)[0] + ".gcode"
    return args.signal.replace(":", "_") + ".gcode"


def cmd_preview(args):
    base = os.path.splitext(args.gcode)[0]
    png = args.png or base + ".preview.png"
    wav = args.wav or base + ".preview.wav"
    dec = preview_file(args.gcode, png, wav, playback_model=not args.raw)
    h = dec["header"]
    print(f"decoded groove: {h['rpm']:.0f} RPM, {dec['t'][-1]:.1f} s, "
          f"recovered peak {dec['recovered_amplitude_mm']*1000:.1f} um "
          f"(commanded {h['amplitude']*1000:.0f} um)")
    print(f"wrote {png}\nwrote {wav}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="openvinyl", description="OpenVinyl v2 groove encoder")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("info", help="print derived groove parameters")
    _add_geometry_args(a)
    a.set_defaults(fn=cmd_info)

    a = sub.add_parser("generate", help="audio -> groove G-code")
    a.add_argument("input", nargs="?", help="input WAV (omit to use --signal)")
    a.add_argument("-o", "--output")
    a.add_argument("--base", help="Bambu Studio exported G-code of the blank disc to splice onto")
    a.add_argument("--base-3mf", help="the same blank's sliced .3mf: also write <output>.3mf, the printer's own format, "
                                      "with the groove inside and its time and filament counted (print it from the "
                                      "printer, not through Bambu Studio, which re-slices it)")
    a.add_argument("--standalone", action="store_true", help="wrap in generic start/end G-code")
    a.add_argument("--z-top", type=float, default=None, help="disc top surface Z if no --base")
    a.add_argument("--center", default=None, help="X,Y of disc centre on the bed")
    a.add_argument("--signal", default="tone:440",
                   help="test signal when no input: silence | tone:F | sweep:F0:F1 | ladder:F0:F1:N")
    a.add_argument("--duration", type=float, default=None, help="seconds of audio to encode")
    a.add_argument("--hpf", type=float, default=300.0)
    a.add_argument("--lpf", type=float, default=None, help="default: nozzle smear limit at inner radius")
    a.add_argument("--ratio", type=float, default=6.0, help="compressor ratio, 1 disables")
    a.add_argument("--target-dr", type=float, default=18.0)
    a.add_argument("--no-dsp", action="store_true")
    a.add_argument("--save-conditioned", help="write the conditioned audio to this WAV")
    _add_geometry_args(a)
    a.set_defaults(fn=cmd_generate)

    a = sub.add_parser("preview", help="decode a groove G-code back into a WAV and PNG")
    a.add_argument("gcode")
    a.add_argument("--png")
    a.add_argument("--wav")
    a.add_argument("--raw", action="store_true", help="skip the first-order playback model")
    a.set_defaults(fn=cmd_preview)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
