# Adapting OpenVinyl to your printer

Every disc that has played so far was printed on one Bambu Lab P1S with a 0.4 mm nozzle and PLA. Nothing in the method is specific to that printer. The encoder writes the groove as G-code directly, so what it needs from a printer is a handful of numbers: how wide a bead is, how hard the head can accelerate, how many short moves per second the firmware keeps up with, and where the bed ends. This guide is how to find those numbers for your machine, check them with one test disc, and then print songs.

Status: the steps below are derived from the encoder and from the P1S results. No other printer has printed a disc yet. If you try one, the test disc's report (step 5) is the useful thing to share.

## What you need

| | Minimum | Why |
|:--|:--|:--|
| Printer | Any FDM printer that runs a G-code file as it is: from an SD card, or uploaded through OctoPrint, Mainsail, Fluidd or the printer's own app | The groove is in the file. Anything that re-slices the file drops it |
| Bed | 180 × 180 mm for a 7 in disc (177.8 mm); 256 × 256 mm for a 10 in disc | The bounds check fails otherwise |
| Nozzle | 0.4 mm | Tested. A 0.2 mm nozzle halves the bead, which doubles the highest frequency the groove holds (below), but has not been printed |
| Filament | PLA, 1.75 mm | Tested |
| Slicer | Any, for the blank disc only. Bambu Studio, OrcaSlicer, PrusaSlicer (and SuperSlicer) and Cura layer markers are recognised | The tool finds the disc's top layer by these markers |
| Turntable | The spine build in `hardware/turntable/` (largest part 195 mm, fits a 200 mm bed), or any turntable that plays 40 RPM and takes a ceramic cartridge | |

## How the groove depends on the printer

The groove is two beads side by side, wiggling together, with the stylus riding the crease between them. Each printer number sets one thing:

| Printer number | Profile field | What it sets |
|:--|:--|:--|
| Bead width $w_e$ | `nozzle_diameter` × `extrusion_width_factor` | The bead spacing, the pitch from turn to turn, and the highest frequency the groove can hold. A feature shorter than one bead along the groove is smeared out, so the limit is $f_{smear} = v / w_e$, where $v$ is the groove speed under the stylus (193 to 352 mm/s on a 7 in disc at 40 RPM: 400 to 730 Hz) |
| Acceleration | `groove_accel_max` | How fast the nozzle may draw each wiggle. The encoder slows the nozzle where the groove bends hardest, $v \le \sqrt{a / \kappa}$, so loud passages print slowly and quiet ones fast. Keep it at or under your firmware's acceleration limit, or the firmware will slow down where the encoder did not plan for it |
| Planner rate | `max_segment_rate`, `min_segment_length` | The groove is thousands of 0.1 mm moves per turn. At 30 mm/s that is 300 moves a second. A planner that can't keep up stutters, and every stutter leaves a blob in the groove |
| Flow | `max_volumetric_flow` | A ceiling on print speed for the bead's cross-section. Rarely binding at groove speeds |
| Speed range | `groove_speed_max`, `groove_speed_min` | The groove pass prints between these. Slower is more accurate and takes longer |
| Bed | `bed_x`, `bed_y`, `bed_margin` | The bounds check |
| Positioning | (not a field) | The smallest wiggle that means anything. Assumed 0.01 mm; the music is cut at 20 to 40 µm peaks |

One effect is worth knowing before you tune anything. The nozzle draws the wiggle far slower than the stylus later plays it: the groove prints at 30 mm/s or less and plays at 193 to 352 mm/s. So a 900 Hz tone is drawn as 77 to 140 Hz of head motion. That is the range where printer frames resonate and where input shaping (Klipper, Prusa) and vibration compensation (Bambu) act. Either one can change the wiggle the nozzle actually draws. The P1S discs were printed with Bambu's default settings, and the effect has not been measured. The test disc measures it: if tones come back weaker than predicted in one band, this is the first suspect, and a lower `groove_speed_max` moves the head motion down and away from the resonance.

## Step by step

### 1. Write a profile

Copy `printers/template.json` to your own file and fill it in. Fields you delete keep the P1S value; a misspelled field is an error, not a silent default.

- `bed_x`, `bed_y`: the printable area.
- `nozzle_diameter`, `filament_diameter`, `layer_height` (0.2 is tested).
- `groove_accel_max`: at or under the acceleration your firmware allows for printing moves. On Marlin, `M503` prints it (`M204 P`). On Klipper it is `max_accel` in `printer.cfg`. 2000 mm/s² is the tested value; most printers allow it.
- `max_segment_rate`: start at 500. If the groove pass stutters (the head pauses, or the bead shows regular blobs), halve it, or keep it and pass `--segment 0.2` to `generate`, which halves the move count. 0.2 mm moves are still well under one bead width, so the groove loses nothing audible.
- `extrusion_width_factor`: leave 1.2 until step 4 measures it.

Check it:

```
cd src
python -m openvinyl info --printer ../printers/yours.json --diameter 177.8 --outer 84 --inner 46 --rpm 40 --amplitude 0.04
```

It prints the bead width, spacing, pitch, turns and the smear and stylus limits for that disc.

### 2. Slice a blank disc

The groove is printed on top of a plain disc, which your own slicer makes with your own start G-code, bed levelling and purge. Make a disc 177.8 mm across and 1.0 mm thick with a 7.24 mm centre hole (it fits the keyed spindle set, `print/spine/plate_6_spindles.3mf`). `printers/disc7_blank.stl` is that disc. Slice it centred on the bed at 0.2 mm layers with your usual PLA settings, and export plain G-code.

The tool reads three things from that file: the height of the top layer, the disc's centre (from the top layer only, so a purge line at the bed edge doesn't pull it off) and whether extrusion is absolute (`M82`) or relative (`M83`). It inserts the groove after the last extrusion and before your end G-code. The tool prints the centre it found; check it is where you put the disc.

### 3. Make the test disc

```
set OV_PRINTER=C:\full\path\to\printers\yours.json
set OV_BASE_GCODE=C:\full\path\to\disc7_yours.gcode
python hardware/turntable/test_disc/make_test_disc.py
```

(`export` instead of `set` on Linux and macOS.) Without `OV_BASE_GCODE` the script slices the blank with Bambu Studio's command line, which only exists on a Bambu setup. The disc holds 31 s of test sections: silence, a 333 Hz reference at both ends, tones from 698 down to 98 Hz, and a 110 Hz amplitude ladder. The script ends with a validation report; every line must pass. It also writes a preview PNG and WAV decoded from the G-code itself.

### 4. Print it, and measure the bead

Send `out/testdisc/test_disc_7in_40rpm.gcode` to the printer as a file and print it. Don't open it in a slicer and export it again. Bambu Studio in particular re-slices a `.gcode.3mf` it opens and drops the groove; a Bambu printer runs the `.gcode.3mf` from the microSD card started on its own screen. After the base disc, the nozzle should trace one continuous spiral for about 11 minutes.

Then measure the bead width. Across one turn, from the outer edge of one bead to the outer edge of the other, the pair is $w_e (1 + s)$ wide, where $s$ is the spacing factor (1.1 by default). Measure it with calipers or a close photo with a scale in it, divide by 2.1, and set `extrusion_width_factor` to that width over the nozzle diameter. Re-run step 3 if it moved by more than a few percent, because the spacing, the pitch and the crease all follow it.

### 5. Play it and read the report

Record a play with the turntable's recorder (`hardware/turntable/adc_capture/capture.py`) and run:

```
python hardware/turntable/test_disc/analyze_test_disc.py take1.wav
```

It reports the platter speed, wow and flutter, each tone's level against the level the encoder predicts for your groove (so the groove's own physics drop out), distortion, where the amplitude ladder stops tracking, and the noise floor.

| What you see | What it means |
|:--|:--|
| Tones follow the prediction | The printer draws what the encoder asks. Go on to songs |
| One band of tones falls short | The head isn't drawing the wiggle at those frequencies: resonance or input shaping (above). Lower `groove_speed_max` and print again |
| The highest tones vanish | Expected for the top one or two: they are above the smear limit. If more go, the bead is wider than the profile says (step 4) |
| The ladder mistracks below 40 µm | Tracking, not the printer: check tracking force and the arm (`docs/BUILD.md`, "Tracking") |
| Regular clicks or a buzz at a fixed rate | Planner stutter blobs. Lower `max_segment_rate` or use `--segment 0.2` |

### 6. Songs

```
python hardware/turntable/test_disc/make_song_disc.py --help
```

`OV_PRINTER` and `OV_BASE_GCODE` work the same way. The song encoder limits the groove to 7° of slope and 70 m/s² of sideways acceleration. Those limits come from what the stylus followed on the test disc, so they belong to the cartridge, not the printer: keep them unless your own test disc's ladder says otherwise.

## Firmware notes

- The groove block uses only `G90`, `M83`, `G92 E0`, `G0`/`G1` with X, Y, Z, E and F, and `M400`, plus comments. If the base disc used absolute extrusion, the block switches back to `M82` and restores the extruder position after itself.
- Progress lines (`M73`) are scaled, and added through the groove, only when the base already has them and a print-time header the tool recognises (Bambu Studio, PrusaSlicer, OrcaSlicer). Otherwise the time on the printer's screen leaves out the groove, about 10 minutes.
- Pressure advance (linear advance on Marlin) matters more here than in a normal print, because the groove's speed changes all along it. Without it the bead is wider where the nozzle slows. Calibrate it for your filament first.
- `generate --standalone` wraps the groove in generic start and end G-code with no base disc. It is for checking motion and planner behaviour, not for a record.

## Printing the turntable on another printer

The STLs are in `hardware/turntable/cad/stl_spine/` (the spine build) and `cad/stl_briefcase/` (the case, which needs a 256 mm bed). The `.3mf` plates in `hardware/turntable/print/` are Bambu Studio projects with the P1S settings: 0.20 mm layers, PLA, 3 walls and 15 % gyroid (5 shells on the platter), 4 walls and 40 % gyroid on the arm plate, no supports, every part already in print orientation.

Small holes are drawn oversize for a 0.4 mm nozzle, which prints them 0.2 to 0.4 mm small: free fits 0.3 mm over, horizontal holes 0.4 mm, self-tapping pilots 0.2 mm. Your printer may differ. Print the fit-test plate first (`print/spine/plate_9_fit_test.3mf`, or `stl_spine/fit_*.stl`): each coupon is cut from the real part in its own orientation, so a fit that fails costs a coupon, not a plate.

## What has been checked

| | State |
|:--|:--|
| Bambu P1S, 0.4 mm, PLA | Test disc and song discs printed and played |
| Splicing a Bambu Studio export | Tested on real exports |
| Splicing PrusaSlicer, OrcaSlicer and Cura exports | Tested on synthetic files only |
| JSON printer profiles | Tested |
| Any other printer | Not yet |
