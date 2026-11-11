# Droplet generator — specification and build notes

No part that can be bought does this job, so the machine carries a home-built generator. This is the least proven part of the whole design. Build and test it on its own before anything else.

## 1. What it has to do

| Quantity | Requirement | Where the number comes from |
|---|---|---|
| Droplet diameter | 0.30 mm (14 nL), ±3 % | print simulation |
| Exit speed at the nozzle | about 1.3 m/s, straight up | the droplet must still be rising at 0.82 m/s when the sound field takes over 36 mm above the nozzle (`replicator/delivery.py`) |
| Speed repeatability | ±1 % for full accuracy; lands up to ±3 % | capture-margin study; the light gate measures each shot and corrects timing |
| Direction | within 1° | 0.04 mm landing error at 1°, 0.4 mm at 2°, lost at 6° |
| Satellites | none | a satellite is a second, uncontrolled droplet |
| Rate | on demand, up to 8 per second, and the first droplet after a pause must match | print timing |
| Liquid | 405 nm acrylate resin, 75–85 mPa·s at 25 °C as sold | thinnest hobby resin found |

## 2. Design chosen

A piezo-disc drop-on-demand generator of the type published by Harris, Liu & Bush (2015) and released as open hardware by the Harris lab ("DropGen", CC-BY-SA). A 35 mm piezo buzzer disc forms one wall of a small liquid-filled chamber; a short voltage pulse bends the disc and pushes one droplet out of the nozzle.

Three changes for this machine:

1. **Nozzle up.** The published generators fire down. One paper fires this type of generator upward into a 40 kHz levitator (Andrade, Camargo & Marzo 2018), with about 1 mm droplets.
2. **Heated nozzle.** Resin at 80 mPa·s is four times too viscous for the range in which this generator has been shown to work. Warmed to roughly 50 °C it should be near 20 mPa·s, which puts it inside that range (Ohnesorge number 0.26 against the published 0.16–0.28, both based on nozzle radius). The viscosity–temperature curve of the resin is an assumption until measured.
3. **Cartridge.** The whole generator slides out of the machine to the front for cleaning.

### Parts

| Part | Specification | Source (status) |
|---|---|---|
| Piezo disc | CUI CEB-35D26, 35 mm, brass side in the liquid | Digi-Key 102-1128-ND (part number read in the published design; price not checked) |
| Chamber body | printed PETG; bore 28.6 mm × 10 mm deep, as in the 2015 paper; drip-tray flange on top | `hardware/print/jet_body.stl` |
| Clamp ring | printed; six M3 × 16 screws into heat-set inserts, tightened evenly | `hardware/print/jet_clamp.stl` |
| Seal | O-ring 1-1/4 in ID × 1/16 in under the disc | McMaster-Carr 9452K113 (number read in the published design) |
| Nozzle | brass M6 3-D printer nozzle, 0.30 mm bore; buy 0.2, 0.25 and 0.4 as well | any 3-D printer parts shop |
| Feed tube + heater | 3-D printer hot-end heat break (M6), aluminium heater block, 24 V heater cartridge, 100 kΩ thermistor | any 3-D printer parts shop |
| Feed-tube seal | O-ring 1/8 in ID × 1/16 in | McMaster-Carr 94115K006 |
| Inlet | 10-24 hose barb with washer, 1/8 in opaque tubing | as in the published design |
| Reservoir | 10 mL syringe barrel, open top, light-tight cap, on a sliding clamp | `hardware/print/res_clamp.stl` |
| Pulse driver | dual H-bridge (L293D as published, or DRV8871) on an adjustable 5–30 V converter | |
| Heater switch | logic-level MOSFET, driven by the hub | |

Cost of these parts is about 60 USD (see the bill of materials; three lines are estimates).

### Dimensions in the machine

- Nozzle tip 82 mm below the build centre, 0.5 mm inside the bottom port. A 3 mm groove in the port plate lets the tip pass when the cartridge slides out.
- The body hangs on a printed dock under the frame; envelope 48 mm diameter × 50 mm, clear of the floor foam by 9 mm.
- Reservoir level 5–15 mm below the nozzle tip. For a 0.3 mm nozzle the meniscus holds for level errors up to ±43 mm (2σ/ρg·a), so this is not critical.

## 3. Drive

The piezo rests at −V. A pulse switches it to +V for a time t and back. Published values for a 0.9 mm nozzle at 30 V: no droplet below 380 µs, one droplet from 380 to 540 µs, satellites above.

Starting procedure (from the open-hardware instructions, adapted):

1. Fill and bleed. Bleeding is done nozzle-up, which is this machine's normal attitude: tap until resin sprays continuously. An air bubble shows as a louder click and no droplets.
2. Lower the reservoir until resin stops creeping out; dab the nozzle dry.
3. Heater to 50 °C. Wait until the block is stable.
4. V = 15 V, t = 50 µs. Raise t in 50 µs steps until one droplet appears. Then adjust V for speed.

## 4. What is not known

| Unknown | Why it matters | How to find out |
|---|---|---|
| Exit speed and its scatter | no paper reports them; ±1 % is the requirement | high-speed strobe images, or two light gates 10 mm apart, 1,000 shots |
| Whether 0.3 mm is reachable | published droplets are 0.5–1.4 mm; the open-hardware page claims "less than 0.2 mm" without data | try nozzles 0.2–0.4 mm |
| Resin at 50 °C | viscosity, and whether it stays stable for hours | measure flow time through the nozzle at 25 / 40 / 50 / 60 °C |
| Stray 405 nm light on the nozzle | resin could gel at the tip between shots | fire a purge shot into the tray every N droplets; add a shutter if needed |
| The longer feed tube | the published design puts the nozzle directly on the chamber; here 22 mm of 3 mm bore are in between | compare with the nozzle screwed straight into a test body |
| Acrylate against PETG and O-rings | swelling or cracking over weeks | soak test |

If the generator cannot reach ±3 % in speed at 0.3 mm, the fallbacks are, in order: a larger droplet (0.4 mm, which the landing simulations were not run for), a commercial micro-valve by quotation, or a different transfer method (the published acoustic printers extrude a hanging drop of 2.5 µL and let the field pull it off; that is 180 times this volume).

## 5. Sources

Read on 2026-11-10 (`out/sourcing/C_droplet_generator.json` has the details and links): Harris, Liu & Bush, Exp. Fluids 56:83 (2015); Harris lab DropGen v3 repository and wiki; Yang et al., Exp. Fluids 23:445 (1997); Amirzadeh thesis, Toronto (2010); Andrade, Camargo & Marzo, Rev. Sci. Instrum. 89:125105 (2018); Chen et al. (AcoustoFab), Adv. Mater. Technol. (2025); Foresti et al., Sci. Adv. 4:eaat1659 (2018).
