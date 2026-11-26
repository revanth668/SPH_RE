# Mars Flyby + Earth Impact Simulation: How the Code Works

**File:** `sph_mars_flyby_impact.py` · **Outputs:** `sph_mars_flyby_impact.gif` (and `.mp4`), plus three stills (`_flyby.png`, `_impact.png`, `_final.png`)

## What happens in the simulation

An impactor of 0.3 Earth masses is on a collision course with the Earth. On day 110 it passes 30,300 km from Mars's centre (8.9 Mars radii) at 9.0 km/s. Mars gets pulled backwards, slows by about 0.86 km/s, and drops onto a smaller, more eccentric orbit. Mars also bends the impactor's path, and the impactor still hits the Earth on day 231 at 11.6 km/s. The two merge into one body of 1.3 Earth masses, which keeps orbiting the Sun.

Everything is at real scale: real masses, real radii, Mars on its real orbit, and the Earth at 1 AU.

| Body | Mass | Radius | Orbit at start |
|---|---|---|---|
| Sun | 1.989 × 10³⁰ kg | 695,700 km | centre of the system |
| Earth | 5.97 × 10²⁴ kg | 6,371 km | circular, 1 AU |
| Mars | 6.42 × 10²³ kg | 3,390 km | real orbit: a = 1.524 AU, e = 0.093 |
| Impactor ("Theia") | 1.79 × 10²⁴ kg (0.3 Earths) | 4,564 km (density 4,500 kg/m³) | crosses Mars's and the Earth's orbits (a ≈ 1.3 AU, e ≈ 0.3 as it reaches the Earth) |

## Results from the actual run

| Quantity | Value |
|---|---|
| Mars flyby | day 109.6, closest approach 30,304 km (8.9 Mars radii), relative speed 9.01 km/s |
| Mars orbit before | a = 1.524 AU, e = 0.093, 1.381–1.666 AU from the Sun, year = 687 days |
| Mars orbit after | a = 1.428 AU, e = 0.117, 1.261–1.595 AU from the Sun, year = 623 days |
| Mars velocity change | 863 m/s (measured against the no-flyby Mars) |
| Mars at the end (3 yr) | 1.25 AU (187 million km) from where it would have been |
| Earth impact | day 231.2, 11.59 km/s, 9.3 × 10³¹ J turned into heat |
| Merged Earth+Impactor | 1.30 Earth masses, radius 7,072 km, orbit a = 1.049 AU, e = 0.063 (0.982–1.115 AU) |
| Conservation | total mass identical to 12 digits; total momentum stays ~10⁻¹⁹ (zero) |
| Cost | aiming 13 s, main run 16 s (60,639 steps), video rendering ~3.5 min |

What these numbers mean:

- **Mars got a shorter year.** The impactor passes behind Mars, so its gravity pulls Mars backwards along its path. A slower planet falls towards the Sun, so Mars's orbit shrinks and its year shortens by 64 days. Mars's closest point to the Sun drops from 1.38 to 1.26 AU.
- **The change is real and permanent.** The kick happens in about a day. Mars then follows its new orbit indefinitely, and the gap from the no-flyby Mars keeps growing (1.25 AU after three years).
- **The impactor was bent too.** Mars deflects the impactor's path by a few degrees. Without the targeting step (below), that alone would make it miss the Earth by hundreds of Earth radii.
- **Impact speed.** The impactor approaches at 6.4 km/s relative to the Earth. The Earth's gravity speeds it up to 11.6 km/s at contact.

## Units

The code uses astronomical units: length in AU (149.6 million km), time in years, mass in solar masses. In these units G = 4π² and the Earth's year is exactly 1. Each body's radius comes from its mass and real 3D bulk density, R = (3m / 4πρ)^(1/3). When two bodies merge their **volumes** add, so the merged radius is (R₁³ + R₂³)^(1/3).

Conversion factors for printing: 1 AU/yr = 4.74 km/s (`KMS`), the energy unit in joules (`JOULE`), 1 AU = 149,597,871 km (`AU_KM`).

## How the code is organised

The script runs top to bottom in five parts.

### Part 0: constants and the physics engine (same core as before)

- **Kernel (`kernel_W`, `kernel_gradW`):** cubic-spline SPH kernel with smoothing length h = 1 Earth radius.
- **`orbital_elements()`:** turns position and velocity relative to the Sun into a, e, perihelion, aphelion, period and orientation. It is used for all the before/after comparisons.
- **`SPHSystem`:** holds every body and advances them with a leapfrog (kick–drift–kick) step. Forces are SPH pressure, artificial viscosity and Newtonian gravity between all pairs. After each step it checks for contact and merges any overlapping pair, conserving mass, momentum and volume.
- **`choose_dt()`:** adaptive time step. It takes the smaller of 2% of any pair's free-fall time and 30% of its time to contact, capped at 5 × 10⁻⁵ yr (~26 minutes). During the flyby and the impact the step drops to minutes, and then seconds.

Changes to the engine in this version:

| Change | Why |
|---|---|
| `exclude_pairs` option (a gravity mask) | While aiming, Earth–impactor gravity is switched off (explained below) |
| `allow_merge` option | Aiming runs must let the impactor pass through the Earth instead of merging |
| `choose_dt()` skips masked pairs | Otherwise the overlapping Earth and impactor would shrink the step to almost zero |
| `sep(a, b)` helper | Distance between two named bodies, used for slow motion and diagnostics |
| Arrays copied on construction (`xp.array`) | Fixes a bug where the simulation overwrote the caller's starting arrays |

### Part 1: designing the scenario

This is the core of the new version. A close flyby is extremely sensitive: shift the impactor by 1 km before the flyby and it can arrive at the Earth thousands of km off. Guessing starting positions would never work, so the code designs the event backwards, in five steps.

1. **Choose the Earth impact** (`HIT_POINT`, `DV_THEIA`). Impact at (1 AU, 0). The Earth moves at 29.8 km/s, and the impactor arrives with an extra (−1.20, +0.60) AU/yr, i.e. 6.4 km/s, falling inward from outside.
2. **Find where the impactor crossed Mars's orbit** (`kepler(..., event_r=R_CROSS)`). Its orbit is traced backwards around the Sun until it is 1.50 AU out. That is 122 days before impact, and that point becomes the flyby location. The flyby is set at t = 0.30 yr, so the impact is planned for t = 0.634 yr.
3. **Put Mars next to that point** (`B_FLYBY`, `MARS_SIDE`). Mars is placed 10 Mars radii to the side of the impactor's path, perpendicular to their relative motion and on the side that makes the impactor pass behind Mars. `state_on_ellipse()` gives Mars the velocity of its real orbit (a = 1.524 AU, e = 0.093) through that point.
4. **Integrate the whole system backwards** (`backward_plan()`). From the impact moment, all four bodies are integrated back to t = 0 with SciPy's high-accuracy DOP853 solver, including the flyby. This produces starting conditions that lead into the planned flyby and impact.
5. **Make Mars's starting orbit the real one** (the loop that prints `Mars pre-flyby orbit`). Step 3 put Mars on its real orbit at the flyby, but the flyby kicks Mars. So in the backward run, Mars's orbit before the flyby came out wrong (a = 1.630 AU on the first try). The code reads Mars's pre-flyby orbit off the backward run and corrects Mars's velocity at the flyby by the difference, then repeats. Six rounds bring it to a = 1.52368, e = 0.0934, i.e. the real Mars.

Gravity between the Earth and the impactor is switched off during design. At the impact instant they occupy the same point, and the force there would be infinite. Leaving it out is safe: in the real run, the Earth's gravity only pulls the impactor further in (gravitational focusing), so aiming at the Earth's centre still gives a hit.

### Part 1e: aiming with the real integrator ("shooting")

The backward design uses one integrator (DOP853) and the simulation uses another (leapfrog). Their tiny differences, around 10⁻⁷ AU, get magnified thousands of times by the flyby, enough to miss the Earth. So `signed_miss(s)` runs the actual simulation code with merges and Earth–impactor gravity off. It returns how far, and on which side, the impactor passes the Earth's centre. The parameter `s` scales the impactor's starting speed by (1 + s).

A secant search adjusts `s` until the miss is under 0.05 Earth radii:

| Iteration | s | Miss (Earth radii) |
|---|---|---|
| 0 | 0 | +0.56 |
| 1 | +1.0 × 10⁻⁶ | +0.99 |
| 2 | −1.30 × 10⁻⁶ | +0.07 |
| 3 | −1.48 × 10⁻⁶ | +0.05 |

The final correction is a change of about 1.5 parts per million in starting speed, a few centimetres per second, which shows how sensitive the flyby is. Interplanetary missions use the same "shooting" method to aim gravity-assist flybys.

### Part 2: the main run and the "ghost" Mars

The main loop runs 3 years with full physics. Each pass picks a step, lands exactly on the next daily snapshot if one is due, advances, merges on contact, and records. While the impactor is within 60 Mars radii of Mars or 40 Earth radii of the Earth, and for ~13 hours after impact, every step is recorded with a 5-minute cap. Those snapshots become the slow-motion parts.

The **ghost Mars** run starts from exactly the same Sun, Earth and Mars but without the impactor, integrated with DOP853. Comparing the two Mars positions shows exactly what the flyby changed.

### Part 3: diagnostics

After the run the code finds the closest Mars approach (`k_fly`) and measures Mars's orbit at t = 0 and 0.05 yr after the flyby. It compares Mars's velocity with the ghost's, checks conservation, and analyses the merger: impact speed, heat released, and the merged body's orbit. All the values in the results table above are printed here.

### Part 4: animation

| Panel | Shows |
|---|---|
| Inner solar system (left, true scale) | Sun, Earth, Mars, impactor with 150-day trails. Dotted: Mars's and the Earth's original orbits. Dashed (after each event): Mars's new orbit and the merged body's orbit. A grey ring marks "Mars if no flyby". |
| Close-up (middle) | Centred on Mars in Mars radii until halfway between flyby and impact, then centred on the Earth in Earth radii. Zooms with the impactor's distance; an arrow points to the Sun. |
| Mars's orbit (top right) | Mars's distance from the Sun, actual vs. no flyby, with a before → after box (a, e, year length). |
| How close is the impactor? (bottom right) | Impactor–Mars and impactor–Earth distance in km, log scale, with flyby and impact marked. |

Normal frames are 3 days apart. Each close encounter gets 140 frames spread evenly in time: the Mars flyby plays about 800× slower and the impact about 450× slower, with the factor shown on screen. The script saves an MP4 through ffmpeg (or a GIF with Pillow if ffmpeg is missing). The GIF was made from the MP4 with ffmpeg at 20 fps.

## Settings to try

| Setting | Default | What it changes |
|---|---|---|
| `B_FLYBY` | 10 Mars radii | Planned miss distance at Mars. Smaller = bigger kick to Mars (below ~1.6 radii they collide). |
| `MARS_SIDE` | +1 | +1: impactor passes behind Mars → Mars slows and moves inward. −1: passes in front → Mars speeds up and moves outward. |
| `M_THEIA` | 0.3 Earths | Impactor mass. The kick to Mars scales roughly in proportion. |
| `DV_THEIA` | (−1.20, 0.60) AU/yr | Impactor's approach to the Earth; sets its orbit and when it crosses Mars's orbit. |
| `R_CROSS` | 1.50 AU | Where along Mars's orbit the flyby happens (must stay in 1.38–1.67 AU). |
| `T_FLYBY` | 0.30 yr | When the flyby happens; the run starts that long before it. |
| `T_TOTAL`, `DT_MAX` | 3 yr, 5 × 10⁻⁵ yr | Run length and largest step. |
| `FRAME_EVERY`, `N_SLOW` | 3 days, 140 | Animation pacing. |

The script needs NumPy, SciPy and Matplotlib, plus ffmpeg for MP4. Run it with `python sph_mars_flyby_impact.py`.

## Limitations

- **One particle per body.** It models orbits, the flyby and the merge, not what happens inside the bodies. There is no debris, no Moon formation, and no tidal effects on Mars.
- **2D.** All orbits lie in one plane. Real Mars is tilted 1.85° to the Earth's orbit, which would make an exact alignment like this even rarer.
- **Only Sun, Earth and Mars.** Jupiter and the other planets are left out. They would change the long-term evolution of Mars's new orbit, but not the flyby itself.
- **Perfect merger.** Any contact merges the bodies completely. A real 11.6 km/s impact would throw out debris.
- **Starting orbits.** The Earth starts on a circular orbit, and Mars's orientation (where its perihelion points) is chosen by the code to fit the encounter rather than taken from the real sky.
