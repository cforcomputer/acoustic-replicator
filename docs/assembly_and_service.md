# Assembly, tools and service

Times below are estimates from the model. Nothing has been built.

## 1. How it comes apart

The machine is built so that the common jobs need no tools and no job needs the cage taken to pieces.

| Job | What to do | Tools | Estimated time |
|---|---|---|---|
| Take the finished part out | open the lid, undo the 2 thumb screws of the chuck, lift the chuck with the pin and the part straight up through the top port | none | 20 s |
| Reach inside the cage (clean transducers, retrieve a dropped part) | take the chuck out; unplug the two cables of the cap at its first board; lift the cap with its six boards straight up. The opening is about 150 mm across. To close, lower the cap into its three V-notches, plug the cables in, run the self-calibration | none | 1 min |
| Clean or swap the droplet generator | open the door, unplug its lead, undo one thumb screw under the dock, slide the cartridge out to the front. The drip tray comes with it | none | 1 min |
| Empty the drip tray only | as above | none | 1 min |
| Replace a transducer board | unplug 2 connectors, remove 4 screws, lift the board out of its pocket | small screwdriver | 5 min |
| Refocus or replace a camera | focus: lens lock ring, from outside. Board: 4 screws. Whole pod: 3 screws | 1.5 / 2 mm hex key | 5 min |
| Reach the electronics | pull off the drawer front (magnetic catches) | none | 10 s |
| Take the whole cage out | unplug the trunk cables at the hub, remove the 3 M5 bolts from inside the drawer, lift the cage with its legs | 4 mm hex key | 10 min |

Each of the tool-free motions was checked in the model for collisions along its whole path (`hardware/motion_report.json`): chuck with a 30 mm part out through the port, cap straight up by 100 mm, cartridge 110 mm to the front, door and lid through 95°, one board 25 mm out of its pocket. No collisions.

What the model does not show: cable slack. Leave 100 mm of slack on the two cap cables and on the cartridge lead.

The cap has no fastener. It weighs about 0.3 kg and sits in three V-shaped notches that locate it. If the machine is to be moved, tape it down.

## 2. Tools

**Printing**

- 3-D printer with a bed of at least 220 × 220 mm and 100 mm height; PETG. The largest part is the lower frame, 177 × 204 × 93 mm.

**Electronics**

- temperature-controlled soldering iron with a fine conical tip (1,000 transducer joints, some 0.5 mm from other pads) and a tip for heat-set inserts (M2.5, M3, M4, M5)
- solder, flux pen, desoldering braid, flush cutters, tweezers, wire strippers
- multimeter
- oscilloscope, two channels, 20 MHz or better (strongly recommended: the 40 kHz drive and the jet pulse cannot be judged without one)
- Raspberry Pi Debug Probe (in the bill of materials) to program the boards

**Mechanical**

- hex keys 1.5, 2, 2.5, 3 and 4 mm (M2, M2.5, M3, M4, M5 socket-head screws)
- Phillips screwdrivers PH1 and PH2 (board screws, wood screws)
- calipers
- small pliers

**Enclosure**

- saw for 6 mm plywood (or have the panels cut), wood glue, clamps, square
- drill with 3, 4.5, 5.5 and 10 mm bits, a 14 mm and a 22 mm hole saw or a step drill
- long knife for the foam, spray adhesive

**Consumables and safety**

- isopropanol, lint-free wipes, nitrile gloves
- thermal paste (Peltier and LED stars), cyanoacrylate or epoxy (diffusers, light gate)
- safety glasses that block 405 nm, for any work with the door interlock bypassed

**For the droplet-generator bench test**

- the machine's own camera and one red strobe on the bench, or any camera with a short flash
- a second light gate (the bill of materials has two)

## 3. Order of assembly

1. Print the 15 parts (`hardware/print/*.stl`). Press in the heat-set inserts: counts by size are in `hardware/fasteners.json`.
2. Build and test the droplet generator on the bench (`docs/droplet_generator.md`). Stop here if it does not work.
3. Order the boards (`hardware/pcb/README.md`). Program one, test it without transducers, then with.
4. Solder the transducers to all boards. Test each board with its own microphone.
5. Build the plywood box, glue in the foam, cut the floor holes (three 22 mm, one 14 mm, three 25 mm for the feet).
6. Bolt the three legs and the dock to the lower frame (3 × M4). Stand it in the box; bolt the feet from inside the drawer (3 × M5).
7. Fit the six lower boards. Fit the middle frame (6 × M3 at the tabs), then the six equatorial boards, which bridge the joint.
8. Fit camera pods, cameras, UV emitters, strobes, line laser, light-gate bracket.
9. Fit the six boards of the cap; set the cap in place.
10. Wire the four board chains and the devices; dress the cables with ties so that none crosses a port.
11. Slide in the generator cartridge; connect the reservoir.
12. Fit the drawer electronics; connect; run the self-calibration.

## 4. Notes on the printed parts

- All printed parts were checked for weak joints: each STL is voxelised, a 0.6–0.8 mm layer is peeled off every surface, and the part must still be one piece. All 15 pass.
- The frame halves, legs and dock locate the boards. Print them slowly, without scaling, and measure one pocket (58.3 mm) before printing the rest.
- The cap and the middle frame are drawn as one piece each. The cap (156 × 162 × 76 mm) prints with the top port down; it needs supports under the plates.
- Print orientation and supports were reasoned about, not sliced.
